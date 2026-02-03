import os
import torch
import torch.distributed as dist
from tp_layers_weight_share import MegatronMLP


def run_test():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 固定输入种子，确保输入数据一致
    torch.manual_seed(10086)

    # 实例化模型 (使用相同的 seed=42)
    model = MegatronMLP(16, 32, world_size, rank, seed=42).to(local_rank)

    # 1. 验证权重指纹 (Fingerprint)
    # 取出第一层的权重求和
    w1_sum = model.dense_h_to_4h.weight.sum().item()

    # 打印出来，我们在终端肉眼对比
    # ColumnParallel: 每个 rank 的权重和应该【不同】（因为切了不同行）
    # 但如果用 all_reduce 加起来，总量应该是固定的
    print(f"[Rank {rank}] Layer1 Weight Sum: {w1_sum:.4f}")

    # 2. 跑一次 Forward
    x = torch.randn(2, 8, 16).to(local_rank)
    y = model(x)

    # 3. 验证输出一致性
    # 因为输入一样，参数虽然被切分但来自于同一个母体
    # 所以所有 Rank 最终拿到的 Output 应该是一模一样的！
    output_checksum = y.sum().item()
    print(f"[Rank {rank}] Final Output Checksum: {output_checksum:.4f}")

    # 在屏障处等待，确保打印整齐
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    run_test()