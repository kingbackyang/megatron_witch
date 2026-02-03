"""
简化版 TP+DP 训练脚本，参考 megatron_transformers/train_tp_dp.py。
默认使用假数据跑通流程，方便后续接入真实数据。
运行示例:
    torchrun --nproc_per_node=4 train_tp_dp_mini.py
"""

import os
import math
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup

from mini_megatron_witch.witch_tp_model import WitchTPModel

# ========= 基础配置 =========
TP_SIZE = int(os.environ.get("TP_SIZE", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 50))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 10))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 64))
USE_FAKE_DATA = os.environ.get("USE_FAKE_DATA", "1") == "1"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/Users/jingruyang/Desktop/research_projects/llm_qat_parallel/witch0_7B_custom",
)


# ========= 初始化与分组 =========
def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return global_rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def setup_groups(tp_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert world_size % tp_size == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"

    my_tp_group = None
    my_dp_group = None

    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_tp_group = group

    for i in range(tp_size):
        ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_dp_group = group

    return my_tp_group, my_dp_group


# ========= 数据集 =========
class RandomTextDataset(IterableDataset):
    """
    生成伪造样本，按 DP 分片。
    每个 DP 副本的所有 TP rank 共享相同的 seed，确保顺序一致。
    """

    def __init__(self, vocab_size, seq_len, shard_id, num_shards, seed=1234, steps=MAX_STEPS):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.seed = seed + shard_id
        self.steps = steps

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        for _ in range(self.steps * 2):  # 多给一些，避免 StopIteration
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
            yield {"input_ids": tokens, "labels": tokens.clone()}


def get_dataloader(tokenizer, global_rank, world_size, tp_size, vocab_size):
    dp_size = world_size // tp_size
    data_shard_id = global_rank // tp_size

    if USE_FAKE_DATA:
        dataset = RandomTextDataset(
            vocab_size=vocab_size,
            seq_len=MAX_LENGTH,
            shard_id=data_shard_id,
            num_shards=dp_size,
            steps=MAX_STEPS,
        )
    else:
        raise NotImplementedError("请接入真实数据集或打开 USE_FAKE_DATA=1 以使用伪造数据。")

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)


# ========= 训练主流程 =========
def train():
    global_rank, world_size, local_rank = setup()
    torch.manual_seed(42 + global_rank)

    tp_group, dp_group = setup_groups(TP_SIZE)
    tp_rank = dist.get_rank(group=tp_group)
    dp_rank = dist.get_rank(group=dp_group)

    if global_rank == 0:
        print(f"[Init] World={world_size}, TP={TP_SIZE}, DP={world_size // TP_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    vocab_size = len(tokenizer)
    if vocab_size % TP_SIZE != 0:
        new_vocab_size = ((vocab_size // TP_SIZE) + 1) * TP_SIZE
        if global_rank == 0:
            print(f"[Vocab] resize {vocab_size} -> {new_vocab_size} for TP divisibility")
        vocab_size = new_vocab_size
        config.vocab_size = vocab_size

    model = WitchTPModel(config, TP_SIZE, tp_rank, tp_group)
    model.to(local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    dataloader = get_dataloader(tokenizer, global_rank, world_size, TP_SIZE, vocab_size)
    data_iter = iter(dataloader)

    model.train()
    for step in range(MAX_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(local_rank)
        labels = batch["labels"].to(local_rank)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        dp_size = world_size // TP_SIZE
        if dp_size > 1:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if global_rank == 0 and step % 5 == 0:
            cur_lr = lr_scheduler.get_last_lr()[0]
            print(f"[Step {step}] loss={loss.item():.4f} lr={cur_lr:.6f}")

    # 保存本 TP 分片（仅 DP rank 0 存一份）
    dist.barrier()
    if dp_rank == 0:
        save_dir = f"./tp_checkpoints/mp_rank_{tp_rank:02d}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "tp_size": TP_SIZE,
                "step": MAX_STEPS,
            },
            os.path.join(save_dir, "model_tp.pt"),
        )
        print(f"[Rank {global_rank}] saved shard to {save_dir}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    train()
