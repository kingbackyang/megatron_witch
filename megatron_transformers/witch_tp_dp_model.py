# witch_tp_model.py (部分更新)
from tp_layers_tp_dp import *  # 引用上面的新类


class WitchTPModel(nn.Module):
    def __init__(self, config, tp_size, tp_rank, tp_group):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group  # 保存组信息

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_size, tp_rank, tp_group
        )

        self.layers = nn.ModuleList([
            WitchTransformerLayer(
                config.hidden_size, config.num_heads, config.intermediate_size,
                tp_size, tp_rank, tp_group
            ) for _ in range(config.num_hidden_layers)
        ])

        self.final_layernorm = nn.LayerNorm(config.hidden_size)

        self.lm_head = VocabParallelHead(
            config.hidden_size, config.vocab_size, tp_size, tp_rank, tp_group
        )

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)
        logits_parallel = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # ... (flatten 逻辑同前) ...
            shift_logits = logits_parallel[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            vocab_start_index = self.tp_rank * (self.config.vocab_size // self.tp_size)

            # 传 tp_group 进去
            loss = vocab_parallel_cross_entropy(
                flat_logits, flat_labels,
                self.tp_size, self.tp_rank, self.tp_group, vocab_start_index
            )

        return {"loss": loss, "logits": logits_parallel}