import torch.nn as nn
import timm


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model(config.model, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.model(x).detach()
