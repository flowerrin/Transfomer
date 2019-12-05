import torch
from torch.autograd import Variable
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LabelSmoothing(nn.Module):
    def __init__(self, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx  # pad = 0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        #self.criterion = nn.NLLLoss(ignore_index=padding_idx)
        #self.criterion = nn.KLDivLoss(reduction="batchmean")
        #'batchmean'
        #"none"
        #"sum"
        
    def forward(self, x, target):
        inputs = x.clone()
        batch, length, vocab_size = inputs.shape
        true_dist = inputs.clone()    # x : [batch, vocab_size, length]  [batch, length, vocab_size]
        target_dist = target.clone()    # target : [batch, length]
        pad = target_dist.eq(self.padding_idx)
        pad = pad.unsqueeze(2).repeat(1, 1, vocab_size)    # [batch, length, vocab_size]
        #pad = pad.unsqueeze(1).repeat(1, vocab_size, 1)    # [batch, vocab_size, length]
        
        e_K = self.smoothing / (vocab_size - 2)
        #e_K = self.smoothing / vocab_size
        true_dist.fill_(e_K)    # e/K
        #true_dist.scatter_(2, target.clone().unsqueeze(2), self.confidence+ e_K)
        true_dist.scatter_(2, target.clone().unsqueeze(2), self.confidence)    # (1-e) * delta + e/K
        
        #true_dist.scatter_(1, target.clone().unsqueeze(1), self.confidence)
        
        inputs = inputs.masked_fill_(pad, 0.0)
        true_dist = true_dist.masked_fill_(pad, 0.0)
        
        criterion = Loss()
        #loss = criterion(inputs, true_dist)
        loss = criterion(inputs.reshape(-1, vocab_size), true_dist.reshape(-1, vocab_size))
        
        #return self.criterion(inputs, true_dist)
        #return self.criterion(inputs.reshape(-1, vocab_size), true_dist.reshape(-1, vocab_size))
        return loss
        
        
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, inputs, targets):
        loss = -1.0 * (inputs * targets).sum(dim=-1)
        loss = loss.sum()
        return loss
