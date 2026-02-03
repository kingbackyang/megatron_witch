import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math


# === 辅助类：带 Group 的并行层 ===

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group  # 关键新增：只在这个组内通信

        self.output_size_per_partition = output_size // tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))

        # 初始化逻辑 (保持不变)
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        master_weight = torch.empty(output_size, input_size)
        nn.init.xavier_normal_(master_weight)
        master_bias = torch.empty(output_size)
        nn.init.zeros_(master_bias)

        start_idx = tp_rank * self.output_size_per_partition
        end_idx = (tp_rank + 1) * self.output_size_per_partition

        with torch.no_grad():
            self.weight.data.copy_(master_weight[start_idx:end_idx, :])
            self.bias.data.copy_(master_bias[start_idx:end_idx])
        torch.set_rng_state(rng_state)

    def forward(self, x):
        # Column Parallel 不需要通信
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group  # 关键

        self.input_size_per_partition = input_size // tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.empty(self.output_size))

        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        master_weight = torch.empty(output_size, input_size)
        nn.init.xavier_normal_(master_weight)
        nn.init.zeros_(self.bias)

        start_idx = tp_rank * self.input_size_per_partition
        end_idx = (tp_rank + 1) * self.input_size_per_partition

        with torch.no_grad():
            self.weight.data.copy_(master_weight[:, start_idx:end_idx])
        torch.set_rng_state(rng_state)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight)
        # ⚠️ 关键修改：只在 TP 组内 All-Reduce
        if self.tp_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)
        output = output_parallel + self.bias
        return output


class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_partition = num_heads // tp_size

        self.qkv_proj = ColumnParallelLinear(hidden_size, 3 * hidden_size, tp_size, tp_rank, tp_group, seed)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, seed)

    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq, 3, self.num_heads_per_partition, self.head_dim)
        q, k, v = qkv[..., 0, :, :].transpose(1, 2), qkv[..., 1, :, :].transpose(1, 2), qkv[..., 2, :, :].transpose(1,
                                                                                                                    2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out_proj(attn_out)


class MegatronMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, intermediate_size, tp_size, tp_rank, tp_group, seed)
        self.activation = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(intermediate_size, hidden_size, tp_size, tp_rank, tp_group, seed)

    def forward(self, x):
        return self.dense_4h_to_h(self.activation(self.dense_h_to_4h(x)))


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

        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        master_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(master_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.data.copy_(master_weight[self.vocab_start_index: self.vocab_end_index])
        torch.set_rng_state(rng_state)

    def forward(self, input_ids):
        input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        masked_input = input_ids - self.vocab_start_index
        masked_input[input_mask] = 0
        output = F.embedding(masked_input, self.weight)
        output[input_mask, :] = 0.0
        # ⚠️ 关键修改：只在 TP 组内 All-Reduce
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output


class VocabParallelHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.linear = ColumnParallelLinear(hidden_size, vocab_size, tp_size, tp_rank, tp_group, seed)

    def forward(self, x):
        return self.linear(x)


def vocab_parallel_cross_entropy(logits_parallel, targets, tp_size, tp_rank, tp_group, vocab_start_index):
    # ⚠️ 关键修改：所有的 all_reduce 都加上 group=tp_group
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


# === WitchTransformerLayer 和 WitchTPModel 也需要更新以传递 tp_group ===

class WitchTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = ParallelAttention(hidden_size, num_heads, tp_size, tp_rank, tp_group, seed)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = MegatronMLP(hidden_size, intermediate_size, tp_size, tp_rank, tp_group, seed)

    def forward(self, x):
        x = x + self.attention(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x