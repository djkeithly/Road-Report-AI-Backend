"""PyTorch model definitions and loading helpers."""

from pathlib import Path

import torch
from torch import nn


class BaselineCrashRiskModel(nn.Module):
    """Simple feed-forward model for binary crash risk probability."""

    def __init__(self, *, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return crash-risk logit."""
        return self.network(features)


def load_model_state(model: nn.Module, *, model_path: str) -> nn.Module:
    """Load model weights when a checkpoint exists, otherwise return model unchanged."""
    checkpoint = Path(model_path)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(state)
    return model
