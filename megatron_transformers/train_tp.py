# 文件名: train_tp.py
import os
import torch
import torch.distributed as dist
import torch.optim as optim
from config import WitchConfig
from witch_tp_model import WitchTPModel


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def get_dummy_batch(batch_size, seq_len, vocab_size, device):
    g = torch.Generator()
    g.manual_seed(1234)  # 必须固定种子，保证所有卡拿到同样的数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
    return input_ids, labels


def train():
    local_rank = setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1. 配置 (Vocab 和 Heads 必须能被 world_size 整除)
    config = WitchConfig(
        vocab_size=1024 * world_size,
        hidden_size=256,
        num_heads=4 * world_size,
        num_hidden_layers=2,
        intermediate_size=1024
    )

    if rank == 0:
        print(f"🔥 TP Training Started. World Size: {world_size}")

    # 2. 模型与优化器
    model = WitchTPModel(config, world_size, rank)
    model.to(local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 3. 训练循环 (跑 20 步)
    model.train()
    for step in range(20):
        input_ids, labels = get_dummy_batch(4, 32, config.vocab_size, local_rank)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0 and rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    # ================= 4. 保存逻辑 (Save Logic) =================
    dist.barrier()  # 等所有人跑完

    save_root = "checkpoints/witch_tp_v1"
    rank_dir = os.path.join(save_root, f"mp_rank_{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)

    save_path = os.path.join(rank_dir, "model_optim_rng.pt")

    if rank == 0:
        print(f"💾 Saving checkpoints to {save_root}...")

    # 保存 State Dict
    state = {
        "model": model.state_dict(),
        "config": config,
        "world_size": world_size
    }
    torch.save(state, save_path)

    print(f"[Rank {rank}] Saved shard to {save_path}")

    cleanup()


if __name__ == "__main__":
    train()