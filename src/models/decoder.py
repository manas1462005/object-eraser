import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv

class Decoder(nn.Module):
    def __init__(self, features):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.double_convs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.double_convs.append(DoubleConv(feature * 2, feature))

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.double_convs[idx](x)

        return x
