import copy
import torch

import torch.nn as nn
from typing import Optional

class Transformer(nn.Module):
    '''
    A Transformer architecture for frame sequences
    '''
    def __init__(self, hidden_dim, out_channels):
        super().__init__()

        self.transformer = nn.Transformer()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

if __name__ == "__main__":
    pass
