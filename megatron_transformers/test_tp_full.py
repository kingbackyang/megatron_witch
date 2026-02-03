import os
import torch
import torch.distributed as dist
from tp_mp_layers import WitchTransformerLayer


def run_test():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.manual_seed(10086)  # 固定输入种子

    # 参数: Hidden=64, Heads=4 (每卡1个), Inter=128
    model = WitchTransformerLayer(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        world_size=world_size,
        rank=rank,
        seed=42
    ).to(local_rank)

    # 验证 Attention 权重的一致性
    # 同样取 QKV 投影的权重和
    qkv_sum = model.attention.qkv_proj.weight.sum().item()
    print(f"[Rank {rank}] Attention QKV Sum: {qkv_sum:.4f}")

    x = torch.randn(2, 8, 64).to(local_rank)
    y = model(x)

    checksum = y.sum().item()
    print(f"[Rank {rank}] Full Layer Output Checksum: {checksum:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_test()