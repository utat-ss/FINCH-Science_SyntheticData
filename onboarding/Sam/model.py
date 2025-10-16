from torch import nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ("dense1", nn.Linear(input_dim, hidden_dim)),
            ("act1", nn.ReLU()),
            ("dense2", nn.Linear(hidden_dim, hidden_dim // 2)),
            ("act2", nn.ReLU()),
            ("dense3", nn.Linear(hidden_dim // 2, output_dim))
        ]))

    def forward(self, x):
        return self.layers(x)
