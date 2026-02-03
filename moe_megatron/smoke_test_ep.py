#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from moe_megatron.moe_tp_model import MoETPModel


def setup():
    backend = os.environ.get("BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, backend


def build_tp_groups(tp_size, world_size):
    rank = dist.get_rank()
    tp_group = None
    for base in range(0, world_size, tp_size):
        ranks = list(range(base, base + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    tp_rank = rank % tp_size
    return tp_group, tp_rank


def build_ep_group(tp_size, ep_size, dp_size):
    rank = dist.get_rank()
    ep_group = None
    mp_size = tp_size * ep_size
    for dp in range(dp_size):
        for tp in range(tp_size):
            ranks = [dp * mp_size + ep * tp_size + tp for ep in range(ep_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                ep_group = group
    ep_rank = (rank // tp_size) % ep_size
    return ep_group, ep_rank


def main():
    rank, world_size, local_rank, backend = setup()

    tp_size = int(os.environ.get("TP_SIZE", 1))
    ep_size = int(os.environ.get("EP_SIZE", 2))
    num_experts = int(os.environ.get("NUM_EXPERTS", 4))
    top_k = int(os.environ.get("TOP_K", 2))
    capacity_factor = float(os.environ.get("CAPACITY_FACTOR", 10.0))
    num_shared_experts = int(os.environ.get("NUM_SHARED_EXPERTS", 1))
    moe_layer_freq = int(os.environ.get("MOE_LAYER_FREQ", 1))

    mp_size = tp_size * ep_size
    assert world_size % mp_size == 0, "world_size must be divisible by TP_SIZE*EP_SIZE"
    dp_size = world_size // mp_size

    tp_group, tp_rank = build_tp_groups(tp_size, world_size)
    ep_group, ep_rank = build_ep_group(tp_size, ep_size, dp_size)

    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")

    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        attention_dropout=0.0,
        num_hidden_layers=2,
        vocab_size=128,
    )

    torch.manual_seed(1234)
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    dist.broadcast(input_ids, src=0)

    ref_model = MoETPModel(
        config=config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_group=tp_group,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        num_shared_experts=num_shared_experts,
        moe_layer_freq=moe_layer_freq,
        aux_loss_coef=0.01,
        z_loss_coef=0.0,
        router_jitter=0.0,
    ).to(device)
    ref_model.eval()

    dist_model = MoETPModel(
        config=config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_group=tp_group,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_group=ep_group,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        num_shared_experts=num_shared_experts,
        moe_layer_freq=moe_layer_freq,
        aux_loss_coef=0.01,
        z_loss_coef=0.0,
        router_jitter=0.0,
    ).to(device)
    dist_model.eval()

    with torch.no_grad():
        ref_out = ref_model(input_ids)["logits"]
        dist_out = dist_model(input_ids)["logits"]

    max_diff = (ref_out - dist_out).abs().max()
    max_diff_tensor = max_diff.detach().clone()
    dist.all_reduce(max_diff_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[SmokeTest] TP={tp_size} EP={ep_size} DP={dp_size} max_diff={max_diff_tensor.item():.6e}")

    assert max_diff_tensor.item() < 1e-4, f"EP output mismatch: max_diff={max_diff_tensor.item()}"

    dist.barrier()
    if rank == 0:
        print("[SmokeTest] EP output matches EP=1 reference.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
