from __future__ import annotations

import torch
from torch import nn

WIDTH = 7
HEIGHT = 6


class Connect4QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * HEIGHT * WIDTH, 64)
        self.fc2 = nn.Linear(64, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
