import copy
import torch

import torch.nn as nn
from typing import Optional

class TemporalTransformer(nn.Module):
    '''
    A Transformer architecture for frame sequences
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
class PositionTransformer(nn.Module):
    '''
    A Transformer architecture for landmark positions
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.row_embed = nn.Parameter(torch.rand(300, in_dim)) #embed dimensions
        self.pos = nn.Parameter()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class SignVideoTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_classes, pos_in, pos_out, temporal_out):
        super().__init__()
        '''
        :params pos_in: int of hidden dim
        '''

        assert hidden_dim % num_heads == 0, 'model dimensions must be divisible by number of heads'
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.position_transformer = PositionTransformer(pos_in, pos_out)
        self.temporal_transformer = TemporalTransformer(pos_out, temporal_out)
        self.linear_class = nn.Linear(temporal_out, num_classes)
        
    def forward(self, inputs):
        inputs = self.spatial_transformer(inputs)
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous() # swap 2nd and 3rd dimensions?
        inputs = self.temporal_transformer(inputs) # (1, temporal_out, ...)
        inputs = inputs.mean(dim=1) # (1, temporal out,)
        res = self.linear_class(inputs) # map (1, temporal out,) to (1, 100) for each class
        return res


if __name__ == "__main__":
    pass
