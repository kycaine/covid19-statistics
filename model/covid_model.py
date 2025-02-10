import torch
import torch.nn as nn

class CovidPredictor(nn.Module):
    def __init__(self):
        super(CovidPredictor, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
