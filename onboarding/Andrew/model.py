import torch.nn as nn

"""
action items for next week:
- train a bigger network (more layers, check for overfitting)
"""

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
