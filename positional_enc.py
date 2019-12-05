import math

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PosEnc(nn.Module):
    def forward(self, inputs):
        batch_size, length, hidden_size = inputs.shape
        in_type = inputs.dtype
        
        hidden_cnt = torch.arange(hidden_size, dtype=in_type, device=device) // 2 * 2
        hidden_mat = hidden_cnt.unsqueeze(0).repeat(length, 1)  #[length, hidden_size]
        hidden_mat = 10000.0 ** (hidden_mat / hidden_size)
        
        #cos(x) = sin(x + pi/2)
        sin_cos = torch.arange(hidden_size, dtype=in_type, device=device) % 2
        sin_cos = sin_cos * (math.pi / 2)
        sin_cos_mat = sin_cos.unsqueeze(0).repeat(length, 1)  #[length, hidden_size]
        
        pos_cnt = torch.arange(length, dtype=in_type, device=device)
        pos_mat = pos_cnt.unsqueeze(1).repeat(1, hidden_size)  #[length, hidden_size]
        
        positional_enc = torch.sin(pos_mat / hidden_mat + sin_cos_mat)
        positional_enc = positional_enc.unsqueeze(0).repeat(batch_size, 1, 1)  #[batch_size, length, hidden_size] 
        
        output = inputs + positional_enc
        #output = torch.where(inputs == 0, inputs, output)  #mask
        return output
