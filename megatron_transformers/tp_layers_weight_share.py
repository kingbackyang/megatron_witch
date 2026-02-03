import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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