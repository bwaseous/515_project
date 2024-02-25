import torch
from torch import nn
from torch.nn import functional as F


class RoadSafetyNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.input_dim: int = 32
        self.dense1 = nn.Linear(self.input_dim, 64)
        self.dense2 = nn.Linear(64,128)
        self.dense3 = nn.Linear(128,128)
        self.dense4 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.tanh(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)

        return F.softmax(x)