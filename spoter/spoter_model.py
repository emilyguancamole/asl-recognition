import copy
import torch

import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])

class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> torch.Tensor:
    
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim): 
        super().__init__()

        # text -> sequences
            # text embedding
        # patches -> images
            # patch embedding
        # frame -> sequences
            # landmark vector

        # define learnable parameters
        # row_embed random Parameter of shape (50, hidden_dim) for row embedding
        # in nlp, input sequence matrix: embedding vector x input words = sequence len x hidden dim
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim)) # 50 for approx number of frames?
        # pos Parameter of shape (1,1,hidden_dim) for positional encoding
        self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        # class_query random Parameter of shape (1, hidden_dim) for attention mechanism
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))

        self.transformer = nn.Transformer(hidden_dim, 10, 6, 6) # define transformer
        self.linear_class = nn.Linear(hidden_dim, num_classes) # define linear map of hidden dimensions to labels

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        '''
        Compute output tensor from  input tensor
        :param inputs: a tensor of shape (frames, 55, 2) representing a video instance
        :return: a tensor of shape (1, 1, 100) for each class
        '''

        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float() ## (frames, 1, 110)
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1) ## (1, 1, 110)
        res = self.linear_class(h) 

        return res


if __name__ == "__main__":
    pass
