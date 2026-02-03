import torch
import torch.nn as nn

from moe_megatron.tp_layers import (
    VocabParallelEmbedding,
    VocabParallelHead,
    DenseTransformerLayer,
    MoETransformerLayer,
    vocab_parallel_cross_entropy,
)


class MoETPModel(nn.Module):
    """
    TP+EP MoE model based on the mini_megatron_witch TP structure with MoE MLP blocks.
    """

    def __init__(
        self,
        config,
        tp_size,
        tp_rank,
        tp_group,
        ep_size,
        ep_rank,
        ep_group,
        num_experts,
        top_k=2,
        capacity_factor=1.25,
        num_shared_experts=1,
        moe_layer_freq=1,
        aux_loss_coef=0.01,
        z_loss_coef=0.0,
        router_jitter=0.0,
        seed=42,
    ):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group

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

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            use_moe = (moe_layer_freq > 0) and (i % moe_layer_freq == 0)
            if use_moe:
                layer = MoETransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=tp_group,
                    ep_size=ep_size,
                    ep_rank=ep_rank,
                    ep_group=ep_group,
                    num_experts=num_experts,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                    activation=activation,
                    num_shared_experts=num_shared_experts,
                    aux_loss_coef=aux_loss_coef,
                    z_loss_coef=z_loss_coef,
                    router_jitter=router_jitter,
                    rms_norm_eps=rms_eps,
                    attention_dropout=attn_dropout,
                    seed=seed + i,
                )
            else:
                layer = DenseTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=tp_group,
                    rms_norm_eps=rms_eps,
                    activation=activation,
                    attention_dropout=attn_dropout,
                    seed=seed + i,
                )
            self.layers.append(layer)

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
        moe_aux = hidden_states.new_tensor(0.0)

        for layer in self.layers:
            if isinstance(layer, MoETransformerLayer):
                hidden_states, layer_aux = layer(hidden_states)
                moe_aux = moe_aux + layer_aux
            else:
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
            lm_loss = vocab_parallel_cross_entropy(
                flat_logits,
                flat_labels,
                self.tp_size,
                self.tp_rank,
                self.tp_group,
                vocab_start_index,
            )
            loss = lm_loss + moe_aux

        return {"loss": loss, "logits": logits_parallel, "moe_aux_loss": moe_aux}
