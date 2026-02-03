import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        self.rank = rank

        # 1. 检查能否整除
        assert output_size % world_size == 0, "Output size must be divisible by world size"
        self.output_size_per_partition = output_size // world_size

        # 2. 初始化权重 (注意形状！)
        # 在 PyTorch 的 Linear 中，权重形状是 (out_features, in_features)
        # 列并行：我们切分的是 out_features (第 0 维)
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size
        ))

        # 初始化 bias
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))

        # 3. 随机初始化 (这里有个坑，后面讲，先随便填数)
        # 不同的 rank 必须初始化不同的权重！
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: [batch, seq, input_size]
        # output shape: [batch, seq, output_size_per_partition]

        # 这一步没有任何通信！大家各算各的。
        output = F.linear(x, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        self.rank = rank

        assert input_size % world_size == 0, "Input size must be divisible by world size"
        self.input_size_per_partition = input_size // world_size

        # 行并行：我们切分的是 in_features (第 1 维)
        # 权重形状: (out_features, in_features_per_partition)
        self.weight = nn.Parameter(torch.empty(
            self.output_size,
            self.input_size_per_partition
        ))

        # Bias 是不需要切分的，因为它加在 reduce 之后
        self.bias = nn.Parameter(torch.empty(self.output_size))

        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x 是上一层的切分输出: [batch, seq, input_size_per_partition]

        # 1. 局部计算
        # 结果形状: [batch, seq, output_size] (此时是 Partial Sum)
        output_parallel = F.linear(x, self.weight)

        # 2. All-Reduce (最核心的一步！)
        # 把所有卡的 Partial Sum 加起来
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        # 3. 加上 Bias
        output = output_parallel + self.bias
        return output


class MegatronMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank):
        super().__init__()

        # 第一层：放大 (H -> 4H)，列并行
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size, rank
        )

        self.activation = nn.GELU()

        # 第二层：缩小 (4H -> H)，行并行
        self.dense_4h_to_h = RowParallelLinear(
            intermediate_size, hidden_size, world_size, rank
        )

    def forward(self, hidden_states):
        # hidden_states: [B, S, H] (完整数据)

        # 1. 切分计算，输出是 [B, S, 2H] (假设 split=2)
        intermediate = self.dense_h_to_4h(hidden_states)

        # 2. 激活 (各卡独立做，不需要通信)
        intermediate = self.activation(intermediate)

        # 3. 聚合计算，内部触发 All-Reduce，输出恢复 [B, S, H]
        output = self.dense_4h_to_h(intermediate)

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