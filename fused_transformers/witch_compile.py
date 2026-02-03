import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.profiler import ProfilerActivity, profile, record_function
from witch_config import WitchConfig

# === 1. 辅助函数与类 (模拟您的环境) ===

class WitchRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是 GQA 的关键，将 KV heads 复制 n_rep 次以匹配 Query heads。
    (B, n_kv_heads, L, D) -> (B, n_heads, L, D)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 简化的 RoPE 实现用于演示
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# === 2. 优化后的 WitchAttention (集成 Flash Attention) ===

class WitchAttention(nn.Module):
    def __init__(self, config: WitchConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = WitchRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = WitchRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Token-level gate logic
        mid_channels = max(4, (config.num_attention_heads * self.head_dim) // 4)
        self.token_gate = nn.Sequential(
            nn.Conv1d(config.num_attention_heads * self.head_dim, mid_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mid_channels, 1, kernel_size=9, padding=4),
        )

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        bsz, q_len, _ = hidden_states.shape
        
        # 1. QKV Projections
        query_stat = self.q_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, -1)
        
        # Split Q into Query and Gate Score
        split_size = self.head_dim * self.num_key_value_groups
        query_part, gate_score_part = torch.split(query_stat, [split_size, split_size], dim=-1)
        
        # Reshape Query
        query_states = query_part.reshape(bsz, q_len, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states).transpose(1, 2) # (B, H, L, D)

        # Gate Score needs to be (B, L, H_dim * H_num)
        gate_score = gate_score_part.reshape(bsz, q_len, -1).contiguous()

        # KV Projections
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE
        cos, sin = position_embeddings
        # 支持 (L, D), (B, L, D), 或已是 (B, 1, L, D)
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 4. SDPA (Flash Attention) 优化核心
        # 使用 F.scaled_dot_product_attention 替代手动 matmul+softmax
        # 这在 torch.compile 下极快
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True if attention_mask is None else False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        # 5. Custom Gating Logic (Witch Logic)
        # Conv1d 需要 (B, C, L)，所以 transpose(1, 2)
        token_gate_input = attn_output.transpose(1, 2) 
        token_gate = torch.sigmoid(self.token_gate(token_gate_input)).transpose(1, 2)
        
        # 融合乘法
        attn_output = attn_output * token_gate * torch.sigmoid(gate_score)
        
        return self.o_proj(attn_output)

# === 3. 运行测试 ===

# 初始化 Config
config = WitchConfig(
    hidden_size=1024, 
    num_attention_heads=16, 
    num_key_value_heads=4  # 启用 GQA
)

# 初始化模型
model = WitchAttention(config, layer_idx=0).cuda().half() # FP16

# 虚拟输入
bsz, seq_len = 2, 64
x = torch.randn(bsz, seq_len, config.hidden_size).cuda().half()
# 模拟 RoPE sin/cos (L, D)
cos = torch.randn(seq_len, config.head_dim).cuda().half() # 简化版形状 (L, D)
sin = torch.randn(seq_len, config.head_dim).cuda().half()

# 编译优化
print("开始编译...")
compiled_model = torch.compile(model, mode="max-autotune")
print("编译完成，运行前向传播...")

# 运行
with torch.no_grad():
    output = compiled_model(x, (cos, sin))
    print(f"Output shape: {output.shape}") # 应该输出 (2, 64, 1024)

# === 4. Profiler ===
hidden_states = x
position_embeddings = (cos, sin)

# 预热 (Warmup) 很重要，避免捕捉到 CUDA 初始化的开销
with torch.no_grad():
    for _ in range(5):
        _ = compiled_model(hidden_states, position_embeddings, attention_mask=None)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/witch_attn"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):  # 运行多次 step
        with record_function("model_inference"):
            compiled_model(hidden_states, position_embeddings, attention_mask=None)
        prof.step()

print("Profiling 完成。请使用 TensorBoard 或 Chrome Tracing 打开 ./log/witch_attn 中的 json 文件。")
