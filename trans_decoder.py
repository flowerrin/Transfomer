from feed_forward import FeedForward
from mul_attention import MultiAttention
from mul_mask_attention import MaskedMultiAttention
from positional_enc import PosEnc
from wrap_residual import Residual
from norm import LayerNorm
from layer_decoder import DecoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size=96, drop_out=0.3):
        super(TransDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.length = 1
        self.hopping = 6
        
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.PosEnc = PosEnc()
        self.drop = nn.Dropout(p=drop_out)
        self.out = nn.Linear(hidden_size, output_size)  # bias=False
        
        # Weight Sharing
        self.out.weight = self.embedding.weight
        
        self.layer = nn.ModuleList([])
        for i in range(self.hopping):
            Attention = MultiAttention(hidden_size, batch_size, drop_out).to(device)
            MaskAttention = MaskedMultiAttention(hidden_size, batch_size, drop_out).to(device)
            FFN = FeedForward(hidden_size, hidden_size*4, hidden_size, drop_out).to(device)
            layer = DecoderLayer(hidden_size, MaskAttention, Attention, FFN, drop_out).to(device)
            self.layer.append(layer)
        
        self.norm = nn.LayerNorm(self.hidden_size).to(device)
        #self.norm = LayerNorm(hidden_size)
            
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, enc_output, flag=1, input_tensor=[]):
        output = self.embedding(input)
        output = output * (self.hidden_size ** 0.5)
        output = self.PosEnc(output)
        output = self.drop(output)
        
        N, T, H = output.shape
        
        for layer in self.layer:
            output = layer(output, enc_output, input, input_tensor, flag)
        
        output = self.norm(output)
        
        output = self.out(output)    
        output = self.softmax(output)
        return output
