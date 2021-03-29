import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = x + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, config, n_classes=50):
        super().__init__()
        self.l1 = nn.Linear(
            in_features=config.input_size, out_features=config.hidden_size
        )
        self.embedding = PositionEmbedding(config)
        self.layers = nn.ModuleList(
            [
                transformers.BertLayer(config.model_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.l2 = nn.Linear(in_features=config.hidden_size, out_features=n_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)[0]

        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.2)
        x = self.l2(x)
        return x
