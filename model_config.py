class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


docred_config = Config(
    bert_path='roberta-base',
    split_k=16,
    W_dim=128,
    attn_dim=128,
    num_class=97,
    transformer_type='roberta'

)
