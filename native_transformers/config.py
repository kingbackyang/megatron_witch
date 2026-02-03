from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Dict, Any


@dataclass
class TrainConfig:
    model_path: str
    data_glob: str
    exp_name: str
    wandb_project: str
    base_output_dir: str = "./experiments"
    batch_size: int = 16
    learning_rate: float = 1e-5
    max_steps: int = 20000
    warmup_steps: int = 100
    max_length: int = 512
    buffer_size: int = 10000
    num_workers: int = 4
    seed: int = 42
    log_every: int = 10
    use_wandb: bool = True
    trust_remote_code: bool = True

    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_output_dir, self.exp_name)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = self.output_dir
        data["checkpoint_dir"] = self.checkpoint_dir
        return data
