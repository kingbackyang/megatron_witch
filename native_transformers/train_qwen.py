from __future__ import annotations

try:
    from native_transformers.config import TrainConfig
    from native_transformers.train import run_training
except ModuleNotFoundError:
    from config import TrainConfig
    from train import run_training


def main() -> None:
    config = TrainConfig(
        model_path="/workspace/models/Qwen3-0.6B",
        data_glob="/data2/megatron_witch_data/*.jsonl",
        exp_name="qwen-0.6B",
        wandb_project="qwen-pretrain",
    )
    run_training(config)


if __name__ == "__main__":
    main()
