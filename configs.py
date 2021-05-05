from dataclasses import dataclass, field
import transformers
import timm


@dataclass
class LstmConfig:
    input_size: int = 134
    hidden_size: int = 256
    num_layers: int = 5
    batch_first: bool = True
    bidirectional: bool = True
    dropout: float = 0.2


@dataclass
class XgbConfig:
    booster: str = "gbtree"
    silent: int = 0
    max_depth: int = 2
    subsample: float = 0.9923301318585108
    colsample_bytree: float = 0.7747027267489391
    reg_lambda: int = 3
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    tree_method: str = "gpu_hist"  ## change it to `hist` if gpu not available


@dataclass
class TransformerConfig:
    size: str
    input_size: int = 134
    max_position_embeddings: int = field(default=256, repr=False)
    layer_norm_eps: float = field(default=1e-12, repr=False)
    hidden_dropout_prob: float = field(default=0.1, repr=False)
    hidden_size: int = field(default=512, repr=False)
    num_attention_heads: int = field(default=8, repr=False)
    num_hidden_layers: int = field(default=4, repr=False)
    model_config: transformers.BertConfig = field(init=False)

    def __post_init__(self):
        assert self.size in ["small", "large"]
        if self.size == "small":
            self.hidden_size = 256
            self.num_attention_heads = 4
            self.num_hidden_layers = 2

        self.model_config = transformers.BertConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
        )


@dataclass
class CnnConfig:
    model: str = "mobilenetv2_100"
    output_dim: int = 1280

    def __post_init__(self):
        available_models = timm.list_models(pretrained=True)
        assert self.model in available_models
