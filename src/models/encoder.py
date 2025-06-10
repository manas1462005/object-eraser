import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv

class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, skip_connections
