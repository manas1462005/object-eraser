import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .blocks import DoubleConv

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        super(UNetGenerator, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = Decoder(features)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)
