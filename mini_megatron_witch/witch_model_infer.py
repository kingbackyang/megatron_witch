import torch
from transformers import AutoConfig, AutoTokenizer, WitchForCausalLM

# 固定随机种子，确保初始化与生成可复现
torch.manual_seed(42)


# 路径保持不变，仅加载配置以便调试网络结构
model_name = "/Users/jingruyang/Desktop/research_projects/llm_qat_parallel/witch0_7B_custom"

# 只加载 config，不读取权重文件
config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True,
)
# 初始化模型（随机权重），便于前向调试
model = WitchForCausalLM(config)
model.eval()
print(model)

# 构造最小 dummy 输入并跑通前向
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True,
)
device = next(model.parameters()).device
inputs = tokenizer("Hello, Witch model!", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print(f"logits shape: {outputs.logits.shape}")

# 尝试生成若干新 token，便于进一步调试
pad_id = tokenizer.pad_token_id
if pad_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"generate output: {decoded}")
