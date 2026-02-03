import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import wandb
import glob


# ================= 配置区 =================
raw_files = glob.glob("/data2/megatron_witch_data/*.jsonl")
print(f"🧐 Found {len(raw_files)} data files.") # 打印出来确认一下
DATA_FILES = {"train": raw_files} # 把列表传进去
# DATA_FILES = {"train": "/data2/megatron_witch_data/*.jsonl"}
MODEL_PATH = "/workspace/models/witch0_7B_custom"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5  # 峰值学习率
MAX_STEPS = 20000
WARMUP_STEPS = 100  # 前 100 步线性增加，后面 Cosine 衰减
MAX_LENGTH = 512
BUFFER_SIZE = 10000
NUM_WORKERS = 4


# 2. 实验名称与路径 (关键修改)
EXP_NAME = "witch-0.5b-v1"
BASE_OUTPUT_DIR = "./experiments" # 你的总输出目录

# 拼接出本次实验的根目录: ./experiments/witch-0.5b-v1
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EXP_NAME)
# 拼接出权重保存的子目录: ./experiments/witch-0.5b-v1/checkpoints
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


def train():
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if global_rank == 0:
        print(f"🔥 Witch Training (Streaming + Cosine LR). World Size: {world_size}")
        print(f"📂 Output Dir: {OUTPUT_DIR}")
        
        # 创建目录结构
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        # 注意：wandb 会自动在 dir 参数下创建 'wandb' 文件夹，所以不需要手动创建 wandb 目录
        
        # 初始化 WandB
        wandb.init(
            project=WANDB_PROJECT,
            name=EXP_NAME,
            # === 关键点: 设置 dir 为 OUTPUT_DIR ===
            # 这样 wandb 的日志就会生成在 ./experiments/witch-0.5b-v1/wandb/ 下
            dir=OUTPUT_DIR, 
            config={
                "model": MODEL_PATH,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "world_size": world_size,
                "max_steps": MAX_STEPS,
                "save_path": CHECKPOINT_DIR
            }
        )
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 2. Data Pipeline (Streaming & Sharding)
    dataset = load_dataset("json", data_files=DATA_FILES, split="train", streaming=True)
    dataset = dataset.shard(num_shards=world_size, index=global_rank)
    dataset = dataset.shuffle(seed=42, buffer_size=BUFFER_SIZE)

    def tokenize(ex):
        try:
            text = ex.get("text", "")
            if not text: return None
            out = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
            out["labels"] = out["input_ids"].copy()
            return out
        except:
            return None

    dataset = dataset.map(tokenize, remove_columns=["text", "input", "output", "instruction"])
    dataset = dataset.filter(lambda x: x is not None)
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. Model Setup
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Optimizer & Scheduler (新增部分)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 这里的 steps 是针对 optimizer.step() 的次数
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS
    )

    # 5. Training Loop
    model.train()
    train_iterator = iter(dataloader)

    if global_rank == 0:
        progress_bar = tqdm(range(MAX_STEPS), desc="Training")

    current_step = 0

    while current_step < MAX_STEPS:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            batch = next(train_iterator)

        batch = {k: v.to(local_rank) for k, v in batch.items()}

        # Forward & Backward
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Optimizer Step
        optimizer.step()

        # === 关键点: Scheduler Step ===
        # 必须在 optimizer 更新后调用
        lr_scheduler.step()

        optimizer.zero_grad()

        if global_rank == 0:
            progress_bar.update(1)
            if current_step % 10 == 0:
                # === 关键点: 获取当前 LR ===
                current_lr = optimizer.param_groups[0]['lr']

                # 在进度条里同时显示 Loss 和 LR
                progress_bar.set_description(
                    f"Step {current_step} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )
                # 发送给 WandB
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": current_step / MAX_STEPS
                }, step=current_step)
                # 如果你也接了 wandb，这里就是：
                # wandb.log({"loss": loss.item(), "lr": current_lr})

        current_step += 1

    if global_rank == 0:
        print("✅ Training Finished!")
        # 保存时记得先 unwap
        # model.module.save_pretrained(...)
        # 1. 取出原始模型 (去除 DDP 包装)
        unwrapped_model = model.module
        
        # 2. 保存模型权重和 Tokenizer 到 checkpoints 子文件夹
        unwrapped_model.save_pretrained(CHECKPOINT_DIR)
        tokenizer.save_pretrained(CHECKPOINT_DIR)
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    train()
