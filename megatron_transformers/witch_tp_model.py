import torch
import torch.nn as nn
from tp_mp_layers import VocabParallelEmbedding, WitchTransformerLayer, VocabParallelHead, vocab_parallel_cross_entropy


class WitchTPModel(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.config = config
        self.world_size = world_size
        self.rank = rank

        # 1. Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, world_size, rank
        )

        # 2. Transformer Layers
        self.layers = nn.ModuleList([
            WitchTransformerLayer(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                world_size, rank
            ) for _ in range(config.num_hidden_layers)
        ])

        self.final_layernorm = nn.LayerNorm(config.hidden_size)

        # 3. LM Head (输出层)
        self.lm_head = VocabParallelHead(
            config.hidden_size, config.vocab_size, world_size, rank
        )

    def forward(self, input_ids, labels=None):
        # input_ids: [Batch, Seq]

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_layernorm(hidden_states)

        # 计算 Logits (切分状态)
        # shape: [Batch, Seq, Vocab / N]
        logits_parallel = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten 维度以便计算 Loss
            # [B, S, V/N] -> [B*S, V/N]
            shift_logits = logits_parallel[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # 计算并行 Loss
            # 获取当前 rank 负责的 vocab 起始位置
            vocab_start_index = self.rank * (self.config.vocab_size // self.world_size)

            loss = vocab_parallel_cross_entropy(
                flat_logits, flat_labels,
                self.world_size, self.rank, vocab_start_index
            )

        return {"loss": loss, "logits": logits_parallel}  # 注意 logits 还是切分的