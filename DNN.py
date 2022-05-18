import numpy as np
import torch
from torch import nn

eps = 1e-8


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.hidden4 = nn.Linear(32, 16)
        self.hidden5 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x.view(-1)
def Loss(y, y_hat):
    Loss_1 = nn.MSELoss(reduction='mean')
    loss = Loss_1(y, y_hat)
    return loss






