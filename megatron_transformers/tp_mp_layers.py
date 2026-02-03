import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank, seed=42):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        self.rank = rank

        assert output_size % world_size == 0
        self.output_size_per_partition = output_size // world_size

        # 1. 定义自己的小权重容器 (先不急着初始化)
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size
        ))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))

        # === 核心修改：一致性初始化 ===
        # 临时固定全局随机种子，确保所有 Rank 生成的 master_weight 是一模一样的
        rng_state = torch.get_rng_state()  # 先保存当前的随机状态，以免影响后续
        torch.manual_seed(seed)

        # A. 生成完整的“母体矩阵” [output_size, input_size]
        master_weight = torch.empty(output_size, input_size)
        nn.init.xavier_normal_(master_weight)

        master_bias = torch.empty(output_size)
        nn.init.zeros_(master_bias)

        # B. 切分：拿走属于我的那一块
        # Column Parallel 是按第 0 维 (行) 切分
        start_idx = rank * self.output_size_per_partition
        end_idx = (rank + 1) * self.output_size_per_partition

        # Copy 进去
        with torch.no_grad():
            self.weight.data.copy_(master_weight[start_idx:end_idx, :])
            self.bias.data.copy_(master_bias[start_idx:end_idx])

        # C. 恢复之前的随机状态 (不影响 Dropout 等后续操作的随机性)
        torch.set_rng_state(rng_state)

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank, seed=42):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        self.rank = rank

        assert input_size % world_size == 0
        self.input_size_per_partition = input_size // world_size

        # 定义自己的小权重
        self.weight = nn.Parameter(torch.empty(
            self.output_size,
            self.input_size_per_partition
        ))
        self.bias = nn.Parameter(torch.empty(self.output_size))

        # === 核心修改：一致性初始化 ===
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        # A. 生成完整的“母体矩阵”
        # 注意：RowParallel 的 shape 是 [output, input]
        master_weight = torch.empty(output_size, input_size)
        nn.init.xavier_normal_(master_weight)

        # Bias 不切分，大家应该拥有一模一样的完整 Bias
        # 所以直接初始化到 self.bias 就行，不用切
        nn.init.zeros_(self.bias)

        # B. 切分
        # Row Parallel 是按第 1 维 (列) 切分
        start_idx = rank * self.input_size_per_partition
        end_idx = (rank + 1) * self.input_size_per_partition

        with torch.no_grad():
            # 注意这里切的是第 1 维
            self.weight.data.copy_(master_weight[:, start_idx:end_idx])

        torch.set_rng_state(rng_state)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight)
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        output = output_parallel + self.bias
        return output


# MegatronMLP 不需要改逻辑，只需要把 seed 传进去
class MegatronMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank, seed=42):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size, rank, seed
        )
        self.activation = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(
            intermediate_size, hidden_size, world_size, rank, seed
        )

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.activation(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output


class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, world_size, rank, seed=42):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.world_size = world_size
        self.rank = rank

        # 1. 检查头数能否被整除
        assert num_heads % world_size == 0, f"Heads ({num_heads}) must be divisible by ranks ({world_size})"
        self.num_heads_per_partition = num_heads // world_size

        # 2. QKV 投影层 (Column Parallel)
        # 输入: hidden_size
        # 输出: 3 * hidden_size (Q + K + V)
        # 因为我们要把 QKV 竖着切，每个 rank 拿走属于自己那些头的 QKV
        self.qkv_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * hidden_size,  # Q, K, V 合并
            world_size=world_size,
            rank=rank,
            seed=seed
        )

        # 3. 输出投影层 (Row Parallel)
        # 也就是 Attention 算完后，把结果映射回去
        self.out_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            world_size=world_size,
            rank=rank,
            seed=seed
        )

    def forward(self, x):
        # x: [batch, seq, hidden]
        batch_size, seq_len, _ = x.shape

        # 1. 计算 QKV (并行)
        # qkv_out: [batch, seq, 3 * hidden / world_size]
        qkv_out = self.qkv_proj(x)

        # 2. 拆分 Q, K, V
        # 现在的 qkv_out 维度是混合的，需要 reshape 才能把 Q, K, V 分开
        # 目标形状: [batch, seq, 3, local_heads, head_dim]
        qkv_out = qkv_out.view(
            batch_size, seq_len,
            3,
            self.num_heads_per_partition,
            self.head_dim
        )

        # 分离 (Permute 为了方便 attention 计算: [batch, heads, seq, dim])
        q = qkv_out[..., 0, :, :].transpose(1, 2)  # [B, local_heads, S, D]
        k = qkv_out[..., 1, :, :].transpose(1, 2)
        v = qkv_out[..., 2, :, :].transpose(1, 2)

        # === 这里可以插入 RoPE (旋转位置编码) ===
        # 为了演示简单，先跳过 RoPE，假设是绝对位置或者 No Position

        # 3. 计算 Scaled Dot Product Attention (各卡算各的，不需要通信！)
        # 因为每个卡只负责自己的几个头，互不干扰
        # output: [B, local_heads, S, D]
        # 使用 Flash Attention 的标准 API
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4. 准备进入 Row Parallel
        # 需要把 [B, local_heads, S, D] 变回 [B, S, local_hidden]
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, -1)
        # 此时维度: [B, S, hidden / world_size]

        # 5. 输出投影 (内部会做 All-Reduce)
        # 这一步会把所有卡的结果加起来，恢复成完整的 [B, S, hidden]
        output = self.out_proj(attn_out)

        return output


class WitchTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, world_size, rank, seed=42):
        super().__init__()

        # Attention Block
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = ParallelAttention(hidden_size, num_heads, world_size, rank, seed)

        # MLP Block
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = MegatronMLP(hidden_size, intermediate_size, world_size, rank, seed)

    def forward(self, x):
        # 1. Attention 部分 (Pre-Norm)
        residual = x
        x_norm = self.input_layernorm(x)
        # 注意：Attention 内部已经处理了 TP，输入输出都是完整的 hidden_size
        x = residual + self.attention(x_norm)

        # 2. MLP 部分 (Pre-Norm)
        residual = x
        x_norm = self.post_attention_layernorm(x)
        # MLP 内部也处理了 TP
        x = residual + self.mlp(x_norm)

        return x


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, world_size, rank, seed=42):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank

        # 1. 确保词表能整除 (通常需要手动 padding 词表大小)
        assert num_embeddings % world_size == 0
        self.vocab_per_partition = num_embeddings // world_size

        # 2. 计算当前 Rank 负责的词表范围
        self.vocab_start_index = rank * self.vocab_per_partition
        self.vocab_end_index = (rank + 1) * self.vocab_per_partition

        # 3. 初始化自己的那部分权重
        self.weight = nn.Parameter(torch.empty(
            self.vocab_per_partition,
            self.embedding_dim
        ))

        # 4. 一致性初始化 (还是那一套老办法)
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        master_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(master_weight, mean=0.0, std=0.02)

        with torch.no_grad():
            self.weight.data.copy_(
                master_weight[self.vocab_start_index: self.vocab_end_index]
            )
        torch.set_rng_state(rng_state)

    def forward(self, input_ids):
        # input_ids: [Batch, Seq]

        # 1. 创建掩码：判断哪些 ID 属于我管
        # 如果 ID < start 或 ID >= end，说明不是我的词，设为 0 (或其他安全值)
        # 为了避免 index out of bound，我们先做一个 offset
        input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)

        # 把不属于我的 ID 临时换成 0，防止查表报错 (反正最后会被 mask 掉)
        masked_input = input_ids - self.vocab_start_index
        masked_input[input_mask] = 0

        # 2. 查表
        output = F.embedding(masked_input, self.weight)

        # 3. 把不属于我的位置清零
        # input_mask 维度是 [B, S]，output 是 [B, S, H]
        # 需要把 mask 扩展到最后一维
        output[input_mask, :] = 0.0

        # 4. All-Reduce 求和
        # 因为只有一个 Rank 会查到非零值，其他 Rank 都是 0，加起来就是对的
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        return output


class VocabParallelHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, world_size, rank, seed=42):
        super().__init__()
        # 这是一个 Column Parallel，因为我们要把 hidden -> vocab 切开
        # 每个 Rank 输出 [Batch, Seq, Vocab / N]
        self.linear = ColumnParallelLinear(
            hidden_size, vocab_size, world_size, rank, seed
        )

    def forward(self, x):
        # x: [B, S, H] -> logits: [B, S, V/N]
        logits_parallel = self.linear(x)
        return logits_parallel


# 简易版并行 Loss (原理：Log-Sum-Exp trick)
# 注意：工业级实现通常会用 CUDA kernel 优化，这里用纯 PyTorch 实现逻辑
def vocab_parallel_cross_entropy(logits_parallel, targets, world_size, rank, vocab_start_index):
    # logits_parallel: [B*S, V_p] (已经 flatten)
    # targets: [B*S]

    # 1. 计算局部的 Max (为了数值稳定性)
    logits_max_local, _ = torch.max(logits_parallel, dim=-1)

    # 2. 计算全局 Max
    logits_max_global = logits_max_local.clone()
    dist.all_reduce(logits_max_global, op=dist.ReduceOp.MAX)

    # 3. 减去 Max (防止 exp 溢出)
    # [B*S, V_p] - [B*S, 1]
    logits_parallel = logits_parallel - logits_max_global.unsqueeze(-1)

    # 4. 计算局部的 Exp Sum
    exp_sum_local = torch.sum(torch.exp(logits_parallel), dim=-1)

    # 5. 计算全局 Exp Sum (分母)
    exp_sum_global = exp_sum_local.clone()
    dist.all_reduce(exp_sum_global, op=dist.ReduceOp.SUM)

    # 6. 计算 Log Softmax 的分母部分: log(sum(exp))
    log_sum_exp = torch.log(exp_sum_global)  # 全局共享

    # 7. 计算分子部分: x[target]
    # 这一步比较绕：只有负责 target 的那个 rank 才有正确的值，其他 rank 都是 0
    # 首先判断 target 是否在我的范围内
    vocab_end_index = vocab_start_index + logits_parallel.size(-1)
    target_mask = (targets >= vocab_start_index) & (targets < vocab_end_index)

    # 把不在范围内的 target 设为 0 (防止索引越界)
    masked_targets = targets - vocab_start_index
    masked_targets[~target_mask] = 0

    # 查出 logits 值
    # gather 只能按 index 查，这里需要 gather 对应 target 的 logit
    logits_at_index = torch.gather(logits_parallel, 1, masked_targets.unsqueeze(-1)).squeeze(-1)

    # 如果 target 不归我管，这一项置 0
    logits_at_index = logits_at_index * target_mask.float()

    # 全局求和：把那个唯一持有真值的 rank 的结果拿过来
    dist.all_reduce(logits_at_index, op=dist.ReduceOp.SUM)

    # 8. 最终 Loss = log(sum(exp)) - logits[target] + max_val (前面减掉的要加回来? 其实 cross entropy 公式里抵消了)
    # CrossEntropy = - log(p) = - (logits[target] - log_sum_exp)
    #              = log_sum_exp - logits[target]
    loss = log_sum_exp - logits_at_index

    return loss.mean()