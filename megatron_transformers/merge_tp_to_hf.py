import torch
import os


def reorder_qkv(tensors, num_heads, head_dim):
    """
    QKV 重排逻辑：从 [Q1,K1,V1, Q2,K2,V2] -> [Q_all, K_all, V_all]
    """
    reshaped_tensors = []
    for t in tensors:
        # t shape: [3 * hidden_per_rank, input_dim]
        input_dim = t.shape[1]
        hidden_per_rank = t.shape[0] // 3
        # 拆分为 [3, heads_per_rank, head_dim, input_dim]
        t = t.view(3, hidden_per_rank // head_dim, head_dim, input_dim)
        reshaped_tensors.append(t)

    # 在 heads (dim=1) 维度拼接 -> [3, total_heads, head_dim, input_dim]
    merged = torch.cat(reshaped_tensors, dim=1)

    # 展平回 [3 * total_hidden, input_dim]
    merged = merged.view(-1, merged.shape[-1])
    return merged


def merge_checkpoints(ckpt_dir, output_file):
    print(f"🔄 Merging shards from {ckpt_dir}...")

    subdirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("mp_rank_")])
    world_size = len(subdirs)
    print(f"   Detected World Size: {world_size}")

    # 1. 加载所有分片
    shards = []
    for rank in range(world_size):
        path = os.path.join(ckpt_dir, f"mp_rank_{rank:02d}", "model_optim_rng.pt")
        print(f"   Loading {path}...")
        # 记得加上 weights_only=False 以防报错
        shards.append(torch.load(path, map_location="cpu", weights_only=False)["model"])

    config = \
    torch.load(os.path.join(ckpt_dir, "mp_rank_00/model_optim_rng.pt"), map_location="cpu", weights_only=False)[
        "config"]
    head_dim = config.hidden_size // config.num_heads

    merged_state = {}
    keys = shards[0].keys()

    for key in keys:
        tensors = [s[key] for s in shards]

        # === 修复点在这里 ===
        # A. Column Parallel (Embed, Head Weight & Bias, MLP Up) -> Dim 0
        # 我把 "lm_head.linear.bias" 加进去了
        if any(x in key for x in
               ["embed_tokens.weight", "lm_head.linear.weight", "lm_head.linear.bias", "mlp.dense_h_to_4h.weight",
                "mlp.dense_h_to_4h.bias"]):
            merged_state[key] = torch.cat(tensors, dim=0)

        # B. Row Parallel (Attention Out, MLP Down) -> Dim 1
        elif any(x in key for x in ["attention.out_proj.weight", "mlp.dense_4h_to_h.weight"]):
            merged_state[key] = torch.cat(tensors, dim=1)

        # C. Attention QKV Weight (重排)
        elif "attention.qkv_proj.weight" in key:
            merged_state[key] = reorder_qkv(tensors, config.num_heads, head_dim)

        # D. Attention QKV Bias (重排)
        elif "attention.qkv_proj.bias" in key:
            t_sq = [t.unsqueeze(-1) for t in tensors]
            merged_state[key] = reorder_qkv(t_sq, config.num_heads, head_dim).squeeze(-1)

        # E. 不切分的层 (LayerNorm, RowBias) -> 取第一个
        else:
            merged_state[key] = tensors[0]

    print(f"💾 Saving merged model to {output_file}...")
    torch.save(merged_state, output_file)
    print("✅ Merge Completed!")


if __name__ == "__main__":
    CKPT_DIR = "checkpoints/witch_tp_v1"
    OUTPUT_FILE = "checkpoints/witch_tp_v1/merged_model.bin"
    merge_checkpoints(CKPT_DIR, OUTPUT_FILE)