# 文件名: verify_merged.py
import torch
import os
from config import WitchConfig
from witch_tp_model import WitchTPModel


def verify():
    ckpt_path = "checkpoints/witch_tp_v1/merged_model.bin"
    if not os.path.exists(ckpt_path):
        print("❌ Error: Merged model file not found!")
        return

    print(f"🧐 Loading merged model from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # 1. 准备单卡 Config (World Size = 1)
    # 注意：这里的 vocab_size 和 num_heads 要填【总数】
    # 之前是 vocab=1024*4=4096, heads=4*4=16
    config = WitchConfig(
        vocab_size=4096,
        hidden_size=256,
        num_heads=16,
        num_hidden_layers=2,
        intermediate_size=1024
    )

    # 2. 初始化“单卡”模型
    # 关键：传入 world_size=1，模型内部就不会切分，而是创建完整的层
    model = WitchTPModel(config, world_size=1, rank=0)

    # 3. 尝试加载权重 (Strict=True)
    # 如果拼接形状有一点点不对，这里就会报错
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Load State Dict Successful! Shapes matched perfectly.")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # 4. 跑一次 Forward
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]

    print(f"🎉 Forward Pass Successful!")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")

    # 5. 最终确认
    expected_shape = (1, 10, 4096)
    if logits.shape == expected_shape:
        print("🏆 验证通过！这是一个标准的 PyTorch 模型权重。")
    else:
        print(f"❌ Shape mismatch! Expected {expected_shape}, got {logits.shape}")


if __name__ == "__main__":
    verify()