from feed_forward import FeedForward
from mul_attention import MultiAttention
from positional_enc import PosEnc
from wrap_residual import Residual
from norm import LayerNorm
from layer_encoder import EncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=96, drop_out=0.3):
        super(TransEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.length = 1
        self.hopping = 6
        
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.PosEnc = PosEnc()
        self.drop = nn.Dropout(p=drop_out)
        
        self.layer = nn.ModuleList([])
        for i in range(self.hopping):
            SelfAttention = MultiAttention(hidden_size, batch_size, drop_out).to(device)
            FFN = FeedForward(hidden_size, hidden_size*4, hidden_size, drop_out).to(device)
            layer = EncoderLayer(hidden_size, SelfAttention, FFN, drop_out).to(device)
            self.layer.append(layer)
            
        self.norm = nn.LayerNorm(self.hidden_size).to(device)
        #self.norm = LayerNorm(hidden_size)
        
    '''
    flag = 0: predict
    flag = 1: learn
    '''
    def forward(self, input, flag=1):
        input_tensor = input.clone()
        output = self.embedding(input)
        output = output * (self.hidden_size ** 0.5)
        output = self.PosEnc(output)
        output = self.drop(output)
        
        N, T, H = output.shape
        
        for layer in self.layer:
            #output = self.layer[i](output, input_tensor, flag)
            output = layer(output, input_tensor, flag)
            
        output = self.norm(output)
        return output
