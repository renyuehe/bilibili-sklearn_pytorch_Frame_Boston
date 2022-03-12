import torch
from torch import nn, optim


class Net01(nn.Module):
    def __init__(self):
        super(Net01, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(13,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,6),
            nn.ReLU(),
            nn.Linear(6,1),
        )

    def forward(self, x):
        # input: NV 结构
        return self.layer(x)