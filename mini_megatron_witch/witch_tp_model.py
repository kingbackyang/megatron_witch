import torch
import torch.nn as nn

from mini_megatron_witch.tp_layers import (
    VocabParallelEmbedding,
    WitchTransformerLayer,
    VocabParallelHead,
    vocab_parallel_cross_entropy,
)


class WitchTPModel(nn.Module):
    """
    Tensor Parallel 版的 Witch 模型。接受 HF Config，使用 tp_layers 中的并行组件。
    """

    def __init__(self, config, tp_size, tp_rank, tp_group, seed=42):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        activation = getattr(config, "hidden_act", "gelu")
        attn_dropout = getattr(config, "attention_dropout", 0.0)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            hidden_size,
            tp_size,
            tp_rank,
            tp_group,
            seed=seed,
        )

        self.layers = nn.ModuleList(
            [
                WitchTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=tp_group,
                    rms_norm_eps=rms_eps,
                    activation=activation,
                    attention_dropout=attn_dropout,
                    seed=seed + i,  # 简单防止所有层权重完全一致
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = nn.LayerNorm(hidden_size, eps=rms_eps)
        self.lm_head = VocabParallelHead(
            hidden_size,
            config.vocab_size,
            tp_size,
            tp_rank,
            tp_group,
            seed=seed,
        )

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)
        logits_parallel = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits_parallel[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            vocab_start_index = self.tp_rank * (self.config.vocab_size // self.tp_size)
            loss = vocab_parallel_cross_entropy(
                flat_logits,
                flat_labels,
                self.tp_size,
                self.tp_rank,
                self.tp_group,
                vocab_start_index,
            )

        return {"loss": loss, "logits": logits_parallel}
