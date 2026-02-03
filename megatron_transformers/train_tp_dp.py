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
import math

# 引入自定义的 TP 组件
from config import WitchConfig
from witch_tp_dp_model import WitchTPModel

# ================= 配置区 =================
# 1. 基础配置
TP_SIZE = 4  # Tensor Parallel 并行度 (例如 8 卡设为 4，则 DP=2)
BATCH_SIZE = 4  # 每张卡的 Batch Size (全局 Batch = BATCH_SIZE * DP_SIZE)
LEARNING_RATE = 1e-4  # TP 模型通常比 HF 模型 LR 稍微大一点
MAX_STEPS = 20000
WARMUP_STEPS = 100
MAX_LENGTH = 512
BUFFER_SIZE = 10000
NUM_WORKERS = 0  # ⚠️ TP 模式下必须为 0，确保同一组内的 Rank 读取数据顺序绝对一致

# 2. 数据路径
raw_files = glob.glob("/data2/megatron_witch_data/*.jsonl")
print(f"🧐 Found {len(raw_files)} data files.")
DATA_FILES = {"train": raw_files}
TOKENIZER_PATH = "/workspace/models/witch0_7B_custom"

# 3. 输出配置
EXP_NAME = "witch-hybrid-0.5b"
BASE_OUTPUT_DIR = "./experiments"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, EXP_NAME)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
WANDB_PROJECT = "witch-pretrain"


# ========================================

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
    """
    创建混合并行所需的通信组
    返回: (my_tp_group, my_dp_group)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dp_size = world_size // tp_size

    assert world_size % tp_size == 0, f"World Size ({world_size}) must be divisible by TP Size ({tp_size})"

    my_tp_group = None
    my_dp_group = None

    # 1. 创建 TP 组 (例如 [0,1,2,3], [4,5,6,7])
    # 这些组内的进程共同计算一个模型层
    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_tp_group = group

    # 2. 创建 DP 组 (例如 [0,4], [1,5], [2,6], [3,7])
    # 这些组内的进程持有完全相同的模型权重切片，但处理不同的数据
    for i in range(tp_size):
        ranks = list(range(i, world_size, tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_dp_group = group

    return my_tp_group, my_dp_group


def get_dataloader(tokenizer, global_rank, world_size, tp_size):
    """
    混合并行的数据加载逻辑
    """
    # 计算 DP 相关的参数
    dp_size = world_size // tp_size
    # data_shard_id 决定了当前进程属于哪个 DP 副本
    # 例如 TP=4: Rank 0,1,2,3 都是 ID 0; Rank 4,5,6,7 都是 ID 1
    data_shard_id = global_rank // tp_size

    dataset = load_dataset("json", data_files=DATA_FILES, split="train", streaming=True)

    # ⚠️ 关键步骤 1: 根据 DP ID 进行分片
    # 这样副本 A 和 副本 B 会拿到不同的数据
    dataset = dataset.shard(num_shards=dp_size, index=data_shard_id)

    # ⚠️ 关键步骤 2: Shuffle Seed 必须固定
    # 确保同一个 DP 组内的 TP Ranks (例如 0,1,2,3) 后的数据顺序完全一致
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

    # num_workers=0 保证顺序确定性
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


def train():
    global_rank, world_size, local_rank = setup()

    # === 1. 设置通信组 ===
    tp_group, dp_group = setup_groups(TP_SIZE)

    # 计算组内 Rank
    tp_rank = dist.get_rank(group=tp_group)  # 0~3
    dp_rank = dist.get_rank(group=dp_group)  # 0~1 (其实也就是 data_shard_id)

    # === 2. 初始化 WandB (仅 Global Rank 0) ===
    if global_rank == 0:
        print(f"🔥 Hybrid Training: World={world_size}, TP={TP_SIZE}, DP={world_size // TP_SIZE}")
        print(f"📂 Output Dir: {OUTPUT_DIR}")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        wandb.init(
            project=WANDB_PROJECT,
            name=EXP_NAME,
            dir=OUTPUT_DIR,
            config={
                "type": "Hybrid Parallel",
                "tp_size": TP_SIZE,
                "dp_size": world_size // TP_SIZE,
                "batch_size_per_device": BATCH_SIZE,
                "lr": LEARNING_RATE
            }
        )

    # === 3. Tokenizer & Config ===
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 处理词表大小 (TP 对齐)
    vocab_size = len(tokenizer)
    if vocab_size % TP_SIZE != 0:
        new_vocab_size = ((vocab_size // TP_SIZE) + 1) * TP_SIZE
        if global_rank == 0:
            print(f"⚠️ Resizing vocab from {vocab_size} to {new_vocab_size} for TP alignment")
        vocab_size = new_vocab_size

    config = WitchConfig(
        vocab_size=vocab_size,
        hidden_size=512,  # 演示用，生产环境请改回 2048/4096
        num_heads=8,  # 确保能被 TP_SIZE 整除
        num_hidden_layers=4,
        intermediate_size=2048
    )

    # === 4. 初始化模型 (传入 tp_group) ===
    model = WitchTPModel(config, TP_SIZE, tp_rank, tp_group)
    model.to(local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS
    )

    # === 5. 数据加载器 ===
    dataloader = get_dataloader(tokenizer, global_rank, world_size, TP_SIZE)
    train_iterator = iter(dataloader)

    model.train()

    if global_rank == 0:
        progress_bar = tqdm(range(MAX_STEPS), desc="Hybrid Training")

    current_step = 0

    while current_step < MAX_STEPS:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            batch = next(train_iterator)

        input_ids = batch["input_ids"].to(local_rank)
        labels = batch["labels"].to(local_rank)

        # Forward (TP 组内自动通信)
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward (计算局部梯度)
        loss.backward()

        # === 6. DP 梯度同步 (手动 All-Reduce) ===
        # 我们需要在持有相同模型权重的 Rank 之间 (DP Group) 平均梯度
        if (world_size // TP_SIZE) > 1:
            for param in model.parameters():
                if param.grad is not None:
                    # 在 DP 组内求平均
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # === 7. 日志 ===
        if global_rank == 0:
            progress_bar.update(1)
            if current_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_description(f"Step {current_step} | Loss: {loss.item():.4f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/progress": current_step / MAX_STEPS
                }, step=current_step)

        current_step += 1

    # === 8. 保存 (每个 Rank 存自己的分片) ===
    if global_rank == 0: print("💾 Training Finished. Saving checkpoints...")
    dist.barrier()

    # 每个 Rank 存到独立的 mp_rank_XX 文件夹
    # 这里的 mp_rank 我们使用 global_rank 来命名文件夹，方便区分
    # 或者为了兼容之前的 merge 脚本，我们可以根据 tp_rank 存，但那样会覆盖
    # 建议结构: checkpoints/dp_rank_00/mp_rank_00/...

    # 为了简化，我们按 global_rank 存，后续合并时需要稍微注意 (或者只合并 DP Rank 0 的那一组)
    # 实际上，合并时只需要任意一个 DP 副本的 TP 分片即可 (因为它们权重同步了)

    # 我们只让 DP Rank 0 的副本 (Global Rank 0~TP_SIZE-1) 进行保存
    # 这样只存一份完整的模型，节省空间
    if dp_rank == 0:
        rank_dir = os.path.join(CHECKPOINT_DIR, f"mp_rank_{tp_rank:02d}")
        os.makedirs(rank_dir, exist_ok=True)
        save_path = os.path.join(rank_dir, "model_optim_rng.pt")

        state = {
            "model": model.state_dict(),
            "config": config,
            "world_size": world_size,  # 这里的 world_size 存下来主要是为了记录
            "tp_size": TP_SIZE,  # 记录 TP 尺寸
            "step": current_step
        }
        torch.save(state, save_path)
        print(f"[Rank {global_rank}] Saved shard {tp_rank} to {save_path}")

    dist.barrier()
    if global_rank == 0:
        print("✅ Checkpoints saved (Only first DP replica).")
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    train()