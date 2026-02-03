import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
import glob
from tqdm import tqdm
import wandb

# 引入我们自己写的 TP 模块
from config import WitchConfig
from witch_tp_model import WitchTPModel

# ================= 配置区 =================
# 1. 数据配置
raw_files = glob.glob("/data2/megatron_witch_data/*.jsonl")
print(f"🧐 Found {len(raw_files)} data files.")
DATA_FILES = {"train": raw_files}

# 2. 模型与训练配置
# 注意：这里主要用 Tokenizer，模型结构由 WitchConfig 定义
TOKENIZER_PATH = "/workspace/models/witch0_7B_custom"
BATCH_SIZE = 4  # TP 模式下 Batch 是全局共享的，显存允许的话可以大点
LEARNING_RATE = 1e-4  # TP 通常需要稍微大一点的学习率，或者保持 1e-5
MAX_STEPS = 20000
WARMUP_STEPS = 100
MAX_LENGTH = 512
BUFFER_SIZE = 10000
NUM_WORKERS = 0  # ⚠️ TP 必须设为 0，保证所有卡读取顺序严格一致

# 3. 输出与 WandB 配置
EXP_NAME = "witch-tp-v1"
BASE_OUTPUT_DIR = "./experiments"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EXP_NAME)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
WANDB_PROJECT = "witch-pretrain"


# ========================================

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def get_dataloader(tokenizer, world_size, rank):
    """
    构造真实数据的 DataLoader
    ⚠️ TP 关键点：绝对不能 shard！所有 Rank 必须拿到完全一样的数据。
    """
    dataset = load_dataset("json", data_files=DATA_FILES, split="train", streaming=True)

    # ⚠️ 这里的 shuffle seed 必须固定且所有 Rank 一致！
    dataset = dataset.shuffle(seed=42, buffer_size=BUFFER_SIZE)

    def tokenize(ex):
        try:
            text = ex.get("text", "")
            if not text: return None
            # 这里的 max_length 必须和 Config 里的 seq_len 一致
            out = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
            out["labels"] = out["input_ids"].copy()
            return out
        except:
            return None

    dataset = dataset.map(tokenize, remove_columns=["text", "input", "output", "instruction"])
    dataset = dataset.filter(lambda x: x is not None)
    dataset = dataset.with_format("torch")

    # num_workers=0 确保主进程读取，顺序绝对确定
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


def train():
    local_rank = setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # === 1. 初始化 WandB & 目录 (仅 Rank 0) ===
    if rank == 0:
        print(f"🔥 Witch TP Training. World Size: {world_size}")
        print(f"📂 Output Dir: {OUTPUT_DIR}")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        wandb.init(
            project=WANDB_PROJECT,
            name=EXP_NAME,
            dir=OUTPUT_DIR,
            config={
                "model_path": TOKENIZER_PATH,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "world_size": world_size,
                "max_steps": MAX_STEPS,
                "type": "Tensor Parallel"
            }
        )

    # === 2. Tokenizer & Config ===
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 准备 TP Config
    vocab_size = len(tokenizer)
    # 确保 Vocab 能被切分
    if vocab_size % world_size != 0:
        new_vocab_size = ((vocab_size // world_size) + 1) * world_size
        if rank == 0:
            print(f"⚠️ Resizing vocab from {vocab_size} to {new_vocab_size} for TP alignment")
        vocab_size = new_vocab_size

    config = WitchConfig(
        vocab_size=vocab_size,
        hidden_size=512,  # 这里用小参数演示，你可以改回 2048 等
        num_heads=8,  # 确保能被 world_size 整除
        num_hidden_layers=4,
        intermediate_size=2048
    )

    # === 3. 初始化 TP 模型 ===
    model = WitchTPModel(config, world_size, rank)
    model.to(local_rank)

    # === 4. 优化器 & Scheduler ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS
    )

    # === 5. 数据加载器 ===
    dataloader = get_dataloader(tokenizer, world_size, rank)
    train_iterator = iter(dataloader)

    model.train()

    if rank == 0:
        progress_bar = tqdm(range(MAX_STEPS), desc="TP Training")

    current_step = 0

    # === 6. 训练循环 ===
    while current_step < MAX_STEPS:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            batch = next(train_iterator)

        # 搬运数据
        input_ids = batch["input_ids"].to(local_rank)
        labels = batch["labels"].to(local_rank)

        # Forward & Backward
        # TP 模型 forward 内部会自动处理 All-Reduce
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # === 7. 日志记录 (仅 Rank 0) ===
        if rank == 0:
            progress_bar.update(1)

            if current_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_description(
                    f"Step {current_step} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )

                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/progress": current_step / MAX_STEPS
                }, step=current_step)

        current_step += 1

    # === 8. 保存逻辑 (TP 分片保存) ===
    if rank == 0: print("💾 Training Finished. Saving checkpoints...")

    # 确保所有进程同步
    dist.barrier()

    # 每个 Rank 存到自己的子文件夹: checkpoints/mp_rank_00/
    rank_dir = os.path.join(CHECKPOINT_DIR, f"mp_rank_{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)

    save_path = os.path.join(rank_dir, "model_optim_rng.pt")

    # 保存内容
    state = {
        "model": model.state_dict(),
        "config": config,
        "world_size": world_size,
        "optimizer": optimizer.state_dict(),  # 这里可以把优化器也存了
        "step": current_step
    }

    torch.save(state, save_path)

    print(f"[Rank {rank}] Saved shard to {save_path}")

    # 等待大家存完
    dist.barrier()

    if rank == 0:
        print("✅ All checkpoints saved!")
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    train()