from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, config, n_classes=50):
        super().__init__()
        config_dict = asdict(config)
        self.lstm = nn.LSTM(**config_dict)
        in_features = (
            config.hidden_size * 2 if config.bidirectional else config.hidden_size
        )
        self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.3)
        x = self.l1(x)
        return x
