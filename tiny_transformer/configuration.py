class ModelConfig():
   
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=20,
        max_position_embeddings=20,
        encoder_layer_nums=6,
        decoder_layer_nums=6,
        num_attention_heads=8,
        hidden_size=512,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layer_nums = encoder_layer_nums
        self.decoder_layer_nums = decoder_layer_nums
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
