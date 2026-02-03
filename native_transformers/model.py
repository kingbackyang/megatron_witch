from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def build_causal_lm(
    model_path: str,
    device: torch.device,
    trust_remote_code: bool = True,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    return model

