from transformers import PretrainedConfig
from typing import List, Optional

class WitchConfig(PretrainedConfig):
    model_type = "witch_model"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,  # 默认使用 GQA (32/8 = 4倍压缩)
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        attention_bias: bool = False, # 现代 LLM 通常不使用 bias
        attention_dropout: float = 0.0,
        sliding_window: int = 4096, # 用于滑动窗口注意力
        layer_types: Optional[List[str]] = None, # 控制每一层的类型
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # 如果未指定 kv_heads，则默认为 num_attention_heads (即 MHA)
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

        # 计算 head_dim，这是 Attention 模块需要的
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 如果没有指定每层的类型，默认全部是 "global" 或标准 attention
        if layer_types is None:
            self.layer_types = ["attention"] * num_hidden_layers
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )