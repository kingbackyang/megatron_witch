"""
Minimal TP+EP+DP MoE training script.
Defaults to fake data for fast pipeline validation.

Example (8 GPUs, TP=2, EP=2, DP=2):
    TP_SIZE=2 EP_SIZE=2 torchrun --nproc_per_node=8 moe_megatron/train_tp_dp_ep_moe.py
"""

import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup

from moe_megatron.moe_tp_model import MoETPModel

# ========= Base config =========
TP_SIZE = int(os.environ.get("TP_SIZE", 2))
EP_SIZE = int(os.environ.get("EP_SIZE", 2))
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

NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 8))
TOP_K = int(os.environ.get("TOP_K", 2))
CAPACITY_FACTOR = float(os.environ.get("CAPACITY_FACTOR", 1.25))
NUM_SHARED_EXPERTS = int(os.environ.get("NUM_SHARED_EXPERTS", 1))
MOE_LAYER_FREQ = int(os.environ.get("MOE_LAYER_FREQ", 1))
ROUTER_AUX_LOSS_COEF = float(os.environ.get("ROUTER_AUX_LOSS_COEF", 0.01))
ROUTER_Z_LOSS_COEF = float(os.environ.get("ROUTER_Z_LOSS_COEF", 0.0))
ROUTER_JITTER = float(os.environ.get("ROUTER_JITTER", 0.0))


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return global_rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def setup_groups(tp_size, ep_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mp_size = tp_size * ep_size
    assert world_size % mp_size == 0, "world_size must be divisible by tp_size * ep_size"
    dp_size = world_size // mp_size

    tp_rank = rank % tp_size
    ep_rank = (rank // tp_size) % ep_size
    dp_rank = rank // mp_size

    tp_group = None
    ep_group = None
    dp_group = None

    for dp in range(dp_size):
        for ep in range(ep_size):
            ranks = [dp * mp_size + ep * tp_size + tp for tp in range(tp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_group = group

    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = [dp * mp_size + ep * tp_size + tp for ep in range(ep_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                ep_group = group

    for ep in range(ep_size):
        for tp in range(tp_size):
            ranks = [dp * mp_size + ep * tp_size + tp for dp in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                dp_group = group

    return tp_group, ep_group, dp_group, tp_rank, ep_rank, dp_rank, dp_size


# ========= Dataset =========
class RandomTextDataset(IterableDataset):
    def __init__(self, vocab_size, seq_len, shard_id, seed=1234, steps=MAX_STEPS):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.shard_id = shard_id
        self.seed = seed + shard_id
        self.steps = steps

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        for _ in range(self.steps * 2):
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
            yield {"input_ids": tokens, "labels": tokens.clone()}


def get_dataloader(global_rank, world_size, tp_size, ep_size, vocab_size):
    dp_size = world_size // (tp_size * ep_size)
    data_shard_id = global_rank // (tp_size * ep_size)

    if USE_FAKE_DATA:
        dataset = RandomTextDataset(
            vocab_size=vocab_size,
            seq_len=MAX_LENGTH,
            shard_id=data_shard_id,
            steps=MAX_STEPS,
        )
    else:
        raise NotImplementedError("Provide a real dataset or set USE_FAKE_DATA=1.")

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)


def sync_gradients(model, dp_group, ep_group, dp_size, ep_size):
    for param in model.parameters():
        if param.grad is None:
            continue
        if dp_size > 1:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
            param.grad.div_(dp_size)
        if ep_size > 1 and not getattr(param, "is_expert", False):
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=ep_group)
            param.grad.div_(ep_size)


# ========= Training loop =========
def train():
    global_rank, world_size, local_rank = setup()
    torch.manual_seed(42 + global_rank)

    tp_group, ep_group, dp_group, tp_rank, ep_rank, dp_rank, dp_size = setup_groups(TP_SIZE, EP_SIZE)

    if global_rank == 0:
        print(
            f"[Init] World={world_size}, TP={TP_SIZE}, EP={EP_SIZE}, DP={dp_size}"
        )

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

    model = MoETPModel(
        config=config,
        tp_size=TP_SIZE,
        tp_rank=tp_rank,
        tp_group=tp_group,
        ep_size=EP_SIZE,
        ep_rank=ep_rank,
        ep_group=ep_group,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        capacity_factor=CAPACITY_FACTOR,
        num_shared_experts=NUM_SHARED_EXPERTS,
        moe_layer_freq=MOE_LAYER_FREQ,
        aux_loss_coef=ROUTER_AUX_LOSS_COEF,
        z_loss_coef=ROUTER_Z_LOSS_COEF,
        router_jitter=ROUTER_JITTER,
    )
    model.to(local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    dataloader = get_dataloader(global_rank, world_size, TP_SIZE, EP_SIZE, vocab_size)
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
        moe_aux = outputs["moe_aux_loss"]
        loss.backward()

        sync_gradients(model, dp_group, ep_group, dp_size, EP_SIZE)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if global_rank == 0 and step % 5 == 0:
            cur_lr = lr_scheduler.get_last_lr()[0]
            print(f"[Step {step}] loss={loss.item():.4f} moe_aux={moe_aux.item():.4f} lr={cur_lr:.6f}")

    dist.barrier()
    if dp_rank == 0:
        save_dir = f"./moe_checkpoints/tp{tp_rank:02d}_ep{ep_rank:02d}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "tp_size": TP_SIZE,
                "ep_size": EP_SIZE,
                "num_experts": NUM_EXPERTS,
                "step": MAX_STEPS,
            },
            os.path.join(save_dir, "model_moe_tp.pt"),
        )
        print(f"[Rank {global_rank}] saved shard to {save_dir}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    train()
