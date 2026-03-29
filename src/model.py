"""简单 MLP，对应 MNIST 28x28 展平输入。"""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
