import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Position-wise Feedforward Neural Network
hidden: 1 (ReLU)
output: 1
'''
class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, drop_out):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.drop(y)
        y = self.l2(y)
        return y
