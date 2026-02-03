from __future__ import annotations

import glob
from typing import List, Sequence

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

try:
    from native_transformers.config import TrainConfig
except ModuleNotFoundError:
    from config import TrainConfig


def resolve_data_files(data_glob: str) -> List[str]:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No data files matched: {data_glob}")
    return files


def build_streaming_dataloader(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    data_files: Sequence[str],
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = load_dataset(
        "json",
        data_files={"train": list(data_files)},
        split="train",
        streaming=True,
    )

    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    dataset = dataset.shuffle(seed=config.seed, buffer_size=config.buffer_size)

    dataset = dataset.filter(lambda ex: bool(ex.get("text")))

    def tokenize(example):
        text = example["text"]
        out = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    remove_columns = None
    if getattr(dataset, "features", None) is not None:
        remove_columns = list(dataset.features.keys())

    dataset = dataset.map(tokenize, remove_columns=remove_columns)
    dataset = dataset.with_format("torch")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
