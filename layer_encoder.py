from feed_forward import FeedForward
from mul_attention import MultiAttention
from positional_enc import PosEnc
from wrap_residual import Residual

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, self_attn, feed_forward, drop_out):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        layer_len = 2
        self.sublayer = nn.ModuleList([])
        for i in range(layer_len):
            res = Residual(hidden_size, drop_out).to(device)
            self.sublayer.append(res)
            
        
    def forward(self, output, input_tensor, flag):
        output = self.sublayer[0](output, lambda x: self.self_attn(x, x, input_tensor, flag))
        output = self.sublayer[1](output, self.feed_forward)
        return output
