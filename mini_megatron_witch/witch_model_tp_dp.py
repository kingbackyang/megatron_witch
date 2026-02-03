"""
TP+DP 版本的 Witch 模型推理/示例训练入口。
使用与 mini_megatron_witch/witch_model_infer.py 相同的 HF 配置，
但通过自定义 TP 层实现张量并行，可在多卡上运行。

运行示例 (4 卡，TP=2, DP=2)：
    TP_SIZE=2 torchrun --nproc_per_node=4 mini_megatron_witch/witch_model_tp_dp.py
"""

import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig

from mini_megatron_witch.witch_tp_model import WitchTPModel


# ===== 基础参数 =====
TP_SIZE = int(os.environ.get("TP_SIZE", 2))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/Users/jingruyang/Desktop/research_projects/llm_qat_parallel/witch0_7B_custom",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 8))
PROMPT = os.environ.get("PROMPT", "Hello, Witch model!")


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
    assert world_size % tp_size == 0, f"world_size {world_size} must be divisible by tp_size {tp_size}"

    my_tp_group, my_dp_group = None, None
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


def gather_vocab_logits(logits_shard, tp_group):
    """将 vocab 方向的分片 logits 拼回完整 logits。"""
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
        logits_shard = outputs["logits"][:, -1, :]  # [b, vocab/TP]
        full_logits = gather_vocab_logits(logits_shard, tp_group)
        next_token = torch.argmax(full_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=1
            )

    return input_ids


def main():
    global_rank, world_size, local_rank = setup()
    torch.manual_seed(42 + global_rank)

    tp_group, dp_group = setup_groups(TP_SIZE)
    tp_rank = dist.get_rank(group=tp_group)
    dp_rank = dist.get_rank(group=dp_group)

    if global_rank == 0:
        print(f"[Init] World={world_size}, TP={TP_SIZE}, DP={world_size // TP_SIZE}")

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

    model = WitchTPModel(config, TP_SIZE, tp_rank, tp_group)
    model.to(local_rank)
    model.eval()

    # 构造输入
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to(local_rank) for k, v in inputs.items()}

    # 前向 logits (不含 loss)
    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    logits_shard = outputs["logits"]
    # 仅 Rank0 打印形状
    if global_rank == 0:
        print(f"logits shard shape: {logits_shard.shape}")

    # 简单生成示例
    with torch.no_grad():
        generated = greedy_generate(
            model,
            tokenizer,
            inputs,
            tp_group=tp_group,
            max_new_tokens=MAX_NEW_TOKENS,
            device=local_rank,
        )

    # 只在每个 DP 副本的 tp_rank==0 上解码打印
    if tp_rank == 0:
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"[DP {dp_rank} | Rank {global_rank}] generate: {text}")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
