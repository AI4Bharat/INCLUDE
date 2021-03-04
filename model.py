import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=134,
            hidden_size=256,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.l1 = nn.Linear(in_features=512, out_features=50)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.3)
        x = self.l1(x)
        return x


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
    def __init__(self):
        super().__init__()
        config = transformers.BertConfig()
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.num_hidden_layers = 4
        config.max_position_embeddings = 169

        self.l1 = nn.Linear(in_features=134, out_features=512)
        self.embedding = PositionEmbedding(config)
        self.layers = nn.ModuleList(
            [transformers.BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.l2 = nn.Linear(in_features=512, out_features=50)

    def forward(self, x):
        x = self.l1(x)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)[0]

        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.2)
        x = self.l2(x)
        return x