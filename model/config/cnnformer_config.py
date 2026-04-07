from transformers import ResNetConfig

class CNNFormerConfig(ResNetConfig):
    def __init__(
        self,
        depths,
        hidden_sizes,
        attention_embed_dim,
        upscaler_kernel_size,
        dropout,
        dims_per_multi_attention_head,
        output_indices=None,
        output_features=None,
        **kwargs
    ):
        assert len(hidden_sizes)==len(depths), "Recieved unequal number of depths and hidden_sizes. Specify a hidden size for reach element in depth"
        super().__init__(depths=depths, hidden_sizes=hidden_sizes, **kwargs)
        self.attention_embed_dim = attention_embed_dim
        self.upscaler_kernel_size = upscaler_kernel_size
        self.dropout = dropout
        self.dims_per_multi_attention_head = dims_per_multi_attention_head

        self.stage_names = ['stem_out'] + [f"layer_{i}_out" for i in range(1, len(depths)+1)]
        self.set_output_features_output_indices(out_features=output_features, out_indices=output_indices)