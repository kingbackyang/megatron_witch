from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

try:
    from native_transformers.config import TrainConfig
    from native_transformers.data_iterator import (
        build_streaming_dataloader,
        resolve_data_files,
    )
    from native_transformers.model import build_causal_lm
except ModuleNotFoundError:
    from config import TrainConfig
    from data_iterator import build_streaming_dataloader, resolve_data_files
    from model import build_causal_lm

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None


def init_distributed() -> Tuple[int, int, int]:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_wandb(config: TrainConfig, world_size: int) -> None:
    if wandb is None:
        raise RuntimeError("wandb is not installed but use_wandb=True")

    wandb.init(
        project=config.wandb_project,
        name=config.exp_name,
        dir=config.output_dir,
        config={
            **config.as_dict(),
            "world_size": world_size,
        },
    )


def run_training(config: TrainConfig) -> None:
    local_rank, global_rank, world_size = init_distributed()
    is_main = global_rank == 0

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    seed = config.seed + global_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        print(f"Output dir: {config.output_dir}")

    if is_main and config.use_wandb:
        setup_wandb(config, world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = resolve_data_files(config.data_glob)
    if is_main:
        print(f"Found {len(data_files)} data files.")

    dataloader = build_streaming_dataloader(
        config=config,
        tokenizer=tokenizer,
        data_files=data_files,
        rank=global_rank,
        world_size=world_size,
    )

    model = build_causal_lm(
        config.model_path,
        device=device,
        trust_remote_code=config.trust_remote_code,
    )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
        )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    model.train()
    data_iter = iter(dataloader)

    progress_bar = tqdm(
        range(config.max_steps),
        desc="Training",
        disable=not is_main,
    )

    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % config.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_description(
                f"Step {step} | Loss {loss.item():.4f} | LR {current_lr:.2e}"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/progress": step / config.max_steps,
                    },
                    step=step,
                )

    if is_main:
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(config.checkpoint_dir)
        tokenizer.save_pretrained(config.checkpoint_dir)
        if config.use_wandb:
            wandb.finish()

    cleanup_distributed()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Native Transformers training")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--wandb-project", default="witch-pretrain")
    parser.add_argument("--base-output-dir", default="./experiments")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        model_path=args.model_path,
        data_glob=args.data_glob,
        exp_name=args.exp_name,
        wandb_project=args.wandb_project,
        base_output_dir=args.base_output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    run_training(parse_args())
