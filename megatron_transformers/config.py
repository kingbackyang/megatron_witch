class WitchConfig:
    def __init__(
        self,
        vocab_size=32000, # 必须能被 world_size 整除
        hidden_size=512,  # 必须能被 num_heads 整除
        num_hidden_layers=4,
        num_heads=8,      # 必须能被 world_size 整除
        intermediate_size=2048,
        max_position_embeddings=512
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings