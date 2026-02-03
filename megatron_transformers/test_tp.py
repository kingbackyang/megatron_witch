import os
import torch
import torch.distributed as dist
from tp_layers import MegatronMLP


def run_test():
    # 1. 初始化环境
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 2. 定义参数
    BATCH, SEQ, HIDDEN, INTERMEDIATE = 2, 8, 16, 32

    # 3. 实例化我们的 TP 模型
    model = MegatronMLP(HIDDEN, INTERMEDIATE, world_size, rank).to(local_rank)

    # 4. 构造输入
    # 注意：所有卡必须拿到【一模一样】的输入
    torch.manual_seed(42)  # 固定种子确保输入一致
    x = torch.randn(BATCH, SEQ, HIDDEN).to(local_rank)

    # 5. 前向传播
    y = model(x)

    # 6. 验证
    print(f"[Rank {rank}] Input shape: {x.shape}")
    print(f"[Rank {rank}] Output shape: {y.shape}")

    # 形状应该变回 [2, 8, 16]
    assert y.shape == (BATCH, SEQ, HIDDEN)
    print(f"✅ Rank {rank}: TP MLP Forward pass successful!")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_test()