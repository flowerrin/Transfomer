from feed_forward import FeedForward
from mul_attention import MultiAttention
from mul_mask_attention import MaskedMultiAttention
from positional_enc import PosEnc
from wrap_residual import Residual

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, self_attn, attn, feed_forward, drop_out):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.attn = attn
        self.feed_forward = feed_forward
        layer_len = 3
        self.sublayer = nn.ModuleList([])
        for i in range(layer_len):
            res = Residual(hidden_size, drop_out).to(device)
            self.sublayer.append(res)
            
        
    def forward(self, output, enc_output, input, input_tensor, flag):
        output = self.sublayer[0](output, lambda x: self.self_attn(x, x, input, flag))
        output = self.sublayer[1](output, lambda x: self.attn(x, enc_output, input_tensor, flag))
        output = self.sublayer[2](output, self.feed_forward)
        return output
