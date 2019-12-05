import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True).to(device)
        std = x.std(-1, keepdim=True).to(device)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
