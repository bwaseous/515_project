import torch
from torch import nn


class CreditDefaultNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.input_dim: int = 33
        self.dense1 = nn.Linear(self.input_dim, 256)
        self.dense2 = nn.Linear(256,256)
        self.dense3 = nn.Linear(256,128)
        self.dense4 = nn.Linear(128,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.sigmoid(self.dense4(x))

        return x