import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """
    Flexible MLP with configurable hidden layers

    Architecture:
    Input: Spectral bands (configurable)
    Hidden: Multiple layers with configurable sizes
    Output: 3 fractions (GV, NPV, Soil)
    """
    def __init__(self, input_size=401, hidden_sizes=[64], dropout_rate=0.2):
        super(MLP, self).__init__()
        
        # Build layers dynamically based on hidden_sizes list
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 3))
        
        self.layers = nn.Sequential(*layers)
        self.architecture = f"{input_size} → {' → '.join(map(str, hidden_sizes))} → 3"

    def forward(self, x):
        return self.layers(x)
    
    def get_architecture(self):
        return self.architecture

