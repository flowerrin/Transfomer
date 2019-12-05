from feed_forward import FeedForward
from mul_attention import MultiAttention
from norm import LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Residual(nn.Module):
    def __init__(self, hidden_size, drop_out):
        super(Residual, self).__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size).to(device)
        #self.norm = LayerNorm(hidden_size)
        self.drop = nn.Dropout(p=drop_out)
        
        
    def forward(self, _input, sublayer):
        tensor = self.norm(_input)
        tensor = sublayer(tensor)
        tensor = self.drop(tensor)
        
        output = _input + tensor
        return output
