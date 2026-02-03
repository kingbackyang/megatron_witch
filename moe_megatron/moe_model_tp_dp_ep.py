"""
TP+EP+DP MoE inference / demo entry.
Example (8 GPUs, TP=2, EP=2, DP=2):
    TP_SIZE=2 EP_SIZE=2 torchrun --nproc_per_node=8 moe_megatron/moe_model_tp_dp_ep.py
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig

from moe_megatron.moe_tp_model import MoETPModel


# ===== Base params =====
TP_SIZE = int(os.environ.get("TP_SIZE", 2))
EP_SIZE = int(os.environ.get("EP_SIZE", 2))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/Users/jingruyang/Desktop/research_projects/llm_qat_parallel/witch0_7B_custom",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 8))
PROMPT = os.environ.get("PROMPT", "Hello, MoE Witch model!")

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


def gather_vocab_logits(logits_shard, tp_group):
    tp_size = dist.get_world_size(tp_group)
    shards = [torch.zeros_like(logits_shard) for _ in range(tp_size)]
    dist.all_gather(shards, logits_shard, group=tp_group)
    return torch.cat(shards, dim=-1)


def greedy_generate(model, tokenizer, inputs, tp_group, max_new_tokens, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids, labels=None)
        logits_shard = outputs["logits"][:, -1, :]
        full_logits = gather_vocab_logits(logits_shard, tp_group)
        next_token = torch.argmax(full_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

    return input_ids


def main():
    global_rank, world_size, local_rank = setup()
    torch.manual_seed(42 + global_rank)

    tp_group, ep_group, dp_group, tp_rank, ep_rank, dp_rank, dp_size = setup_groups(TP_SIZE, EP_SIZE)

    if global_rank == 0:
        print(f"[Init] World={world_size}, TP={TP_SIZE}, EP={EP_SIZE}, DP={dp_size}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
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
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(local_rank) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    logits_shard = outputs["logits"]
    if global_rank == 0:
        print(f"logits shard shape: {logits_shard.shape}")

    with torch.no_grad():
        generated = greedy_generate(
            model,
            tokenizer,
            inputs,
            tp_group=tp_group,
            max_new_tokens=MAX_NEW_TOKENS,
            device=local_rank,
        )

    if tp_rank == 0:
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"[DP {dp_rank} | Rank {global_rank}] generate: {text}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
