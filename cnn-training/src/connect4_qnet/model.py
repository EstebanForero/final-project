from __future__ import annotations

import torch
from torch import nn

WIDTH = 7
HEIGHT = 6


class Connect4QNet(nn.Module):
    """Legacy action-value model (7 outputs). Kept for loading old weights only."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * HEIGHT * WIDTH, 64)
        self.fc2   = nn.Linear(64, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Connect4VNet(nn.Module):
    """
    Position-value model: outputs a single V-value ∈ (-1, 1) for the current player.

    Identical feature extraction to Connect4QNet but the head predicts
    'how good is this position?' rather than 'how good is each column?'.
    Trained with game outcomes (win=+1, draw=0, loss=-1) instead of MCTS Q-values.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * HEIGHT * WIDTH, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)).squeeze(1)  # (batch,)
