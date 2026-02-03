from __future__ import annotations

try:
    from native_transformers.config import TrainConfig
    from native_transformers.train import run_training
except ModuleNotFoundError:
    from config import TrainConfig
    from train import run_training


def main() -> None:
    config = TrainConfig(
        model_path="/workspace/models/witch0_7B_custom",
        data_glob="/data2/megatron_witch_data/*.jsonl",
        exp_name="witch-0.5b-v1",
        wandb_project="witch-pretrain",
    )
    run_training(config)


if __name__ == "__main__":
    main()
