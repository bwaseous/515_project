import torch
from torch import nn
from torch.nn import functional as F


class CreditDefaultNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.input_dim: int = 33
        self.dense1 = nn.Linear(self.input_dim, 128)
        self.dense2 = nn.Linear(128,128)
        self.dense3 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.sigmoid(self.dense3(x))

        return x