from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

try:
    from native_transformers.model import build_causal_lm
except ModuleNotFoundError:
    from model import build_causal_lm


def generate_text(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_causal_lm(model_path, device=device, trust_remote_code=True)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Transformers inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = generate_text(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output)
