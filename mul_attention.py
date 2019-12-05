import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiAttention(nn.Module):
    def __init__(self, hidden_size, batch_size, drop_out=0.3):
        super(MultiAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = 8
        self.batch = batch_size
        self.drop = nn.Dropout(p=drop_out)
        self.head_hidden = int(hidden_size / self.head_num)
        
        self.attn_in_q = nn.Linear(hidden_size, hidden_size)
        self.attn_in_k = nn.Linear(hidden_size, hidden_size)
        self.attn_in_v = nn.Linear(hidden_size, hidden_size)        
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        
        self.attn_softmax = nn.Softmax(dim=2)
        
    def forward(self, output, enc_output, enc_input=[], flag=1):
        self.batch, _ , _ = output.shape
        q = self.attn_in_q(output)
        k = self.attn_in_k(enc_output)
        v = self.attn_in_v(enc_output)
        
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)
        
        head_N, T, H = k.shape
        dec_head_N, dec_T, dec_H = q.shape
        
        #[batch_size, head_num, length, hidden_dim/head_num]
        #tmp_tensor = torch.zeros(dec_N, dec_T, T, device=device)
        PAD_num = 0
        
        #attn_weight_n = torch.matmul(q, k.transpose(-2, -1))  #[batch_size, head_num, q_length, k_length]
        attn_weight_n = torch.bmm(q, k.transpose(1, 2))  #[batch_size*head_num, q_length, k_length]
        
        # scaled dot-product
        attn_weight_n *= (H ** -0.5)
        
        pad_weight = enc_input.eq(PAD_num)
        pad_weight = pad_weight.unsqueeze(1).repeat(1, dec_T, 1)
        #pad_weight = pad_weight.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        pad_weight = pad_weight.repeat(self.head_num, 1, 1)
        attn_weight_n = attn_weight_n.masked_fill_(pad_weight, -float('inf'))
        
        attn_weight = self.attn_softmax(attn_weight_n)
        attn_weight = self.drop(attn_weight)
        
        #contexts = torch.matmul(attn_weight, v)
        contexts = torch.bmm(attn_weight, v)
        contexts = contexts.view(self.head_num, self.batch, dec_T, dec_H)
        contexts = self._combine(contexts)  #[batch_size, length, self.hidden_size]
        contexts = self.attn_out(contexts)
        return contexts
    
    def _split(self, x):
        batch_size, length, hidden_size = x.shape
        #x = x.reshape(batch_size, length, self.head_num, int(self.hidden_size / self.head_num))
        x = x.view(batch_size, length, self.head_num, int(self.hidden_size / self.head_num))
        #x = x.transpose(1, 2)
        x = x.permute(2, 0, 1, 3).contiguous().view(-1, length, int(self.hidden_size / self.head_num))
        return x

    def _combine(self, x):
        #batch_size, head_num, length, hidden_size = x.shape
        head_num, batch_size, length, hidden_size = x.shape
        #x = x.transpose(1, 2)
        x = x.permute(1, 2, 0, 3)
        #x = x.reshape(batch_size, length, self.hidden_size)
        x = x.contiguous().view(batch_size, length, self.hidden_size)
        return x
