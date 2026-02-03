import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _init_master_weight(weight_shape, seed=42, init_fn=None):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    master = torch.empty(*weight_shape)
    if init_fn is None:
        nn.init.xavier_normal_(master)
    else:
        init_fn(master)
    torch.set_rng_state(rng_state)
    return master


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size, tp_rank, tp_group, bias=True, seed=42):
        super().__init__()
        self.output_size_per_partition = output_size // tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None

        master_weight = _init_master_weight((output_size, input_size), seed)
        start_idx = tp_rank * self.output_size_per_partition
        end_idx = (tp_rank + 1) * self.output_size_per_partition
        with torch.no_grad():
            self.weight.copy_(master_weight[start_idx:end_idx, :])
            if bias:
                master_bias = torch.zeros(output_size)
                self.bias.copy_(master_bias[start_idx:end_idx])

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size, tp_rank, tp_group, bias=True, seed=42):
        super().__init__()
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.input_size_per_partition = input_size // tp_size

        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None

        master_weight = _init_master_weight((output_size, input_size), seed)
        start_idx = tp_rank * self.input_size_per_partition
        end_idx = (tp_rank + 1) * self.input_size_per_partition
        with torch.no_grad():
            self.weight.copy_(master_weight[:, start_idx:end_idx])

    def forward(self, x):
        out_parallel = F.linear(x, self.weight)
        if self.tp_size > 1:
            dist.all_reduce(out_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out_parallel = out_parallel + self.bias
        return out_parallel


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        self.vocab_per_partition = num_embeddings // tp_size
        self.vocab_start_index = tp_rank * self.vocab_per_partition
        self.vocab_end_index = (tp_rank + 1) * self.vocab_per_partition

        self.weight = nn.Parameter(torch.empty(self.vocab_per_partition, embedding_dim))

        master_weight = _init_master_weight((num_embeddings, embedding_dim), seed, init_fn=lambda w: nn.init.normal_(w, mean=0.0, std=0.02))
        with torch.no_grad():
            self.weight.copy_(master_weight[self.vocab_start_index:self.vocab_end_index])

    def forward(self, input_ids):
        input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        masked = input_ids - self.vocab_start_index
        masked[input_mask] = 0
        output = F.embedding(masked, self.weight)
        output[input_mask, :] = 0.0
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output


class VocabParallelHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.linear = ColumnParallelLinear(hidden_size, vocab_size, tp_size, tp_rank, tp_group, seed=seed)

    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).type_as(x)


class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, tp_size, tp_rank, tp_group, attention_dropout=0.0, seed=42):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_partition = num_heads // tp_size

        self.qkv_proj = ColumnParallelLinear(hidden_size, 3 * hidden_size, tp_size, tp_rank, tp_group, seed=seed)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, seed=seed)
        self.attention_dropout = attention_dropout

    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)  # [b, s, 3 * hidden/ tp]
        qkv = qkv.view(batch, seq, 3, self.num_heads_per_partition, self.head_dim)

        q = qkv[:, :, 0].transpose(1, 2)  # [b, h, s, d]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attention_dropout, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out_proj(attn_out)


class MegatronMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, tp_size, tp_rank, tp_group, activation="gelu", seed=42):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, intermediate_size, tp_size, tp_rank, tp_group, seed=seed)
        self.dense_4h_to_h = RowParallelLinear(intermediate_size, hidden_size, tp_size, tp_rank, tp_group, seed=seed)
        if activation == "silu":
            self.activation = F.silu
        else:
            self.activation = F.gelu

    def forward(self, x):
        return self.dense_4h_to_h(self.activation(self.dense_h_to_4h(x)))


def vocab_parallel_cross_entropy(logits_parallel, targets, tp_size, tp_rank, tp_group, vocab_start_index):
    logits_max_local, _ = torch.max(logits_parallel, dim=-1)
    logits_max_global = logits_max_local.clone()
    dist.all_reduce(logits_max_global, op=dist.ReduceOp.MAX, group=tp_group)

    logits_parallel = logits_parallel - logits_max_global.unsqueeze(-1)
    exp_sum_local = torch.sum(torch.exp(logits_parallel), dim=-1)
    exp_sum_global = exp_sum_local.clone()
    dist.all_reduce(exp_sum_global, op=dist.ReduceOp.SUM, group=tp_group)

    log_sum_exp = torch.log(exp_sum_global)

    vocab_end_index = vocab_start_index + logits_parallel.size(-1)
    target_mask = (targets >= vocab_start_index) & (targets < vocab_end_index)
    masked_targets = targets - vocab_start_index
    masked_targets[~target_mask] = 0
    logits_at_index = torch.gather(logits_parallel, 1, masked_targets.unsqueeze(-1)).squeeze(-1)
    logits_at_index = logits_at_index * target_mask.float()
    dist.all_reduce(logits_at_index, op=dist.ReduceOp.SUM, group=tp_group)

    loss = log_sum_exp - logits_at_index
    return loss.mean()


class WitchTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, tp_size, tp_rank, tp_group, rms_norm_eps=1e-6, activation="gelu", attention_dropout=0.0, seed=42):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = ParallelAttention(hidden_size, num_heads, tp_size, tp_rank, tp_group, attention_dropout, seed)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MegatronMLP(hidden_size, intermediate_size, tp_size, tp_rank, tp_group, activation, seed)

    def forward(self, x):
        x = x + self.attention(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
