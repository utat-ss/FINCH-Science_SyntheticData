import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """
    MLP with just 1 hidden layer 

    Architecture:
    Input: 401 Spectral bands
    Hidden: 64 Neurons
    Output: 3 fractions (Npv, Soil, gv)
    """
    def __init__(self, input_size=401, hidden_size=64, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3))

    def forward(self, x):
        return self.layers(x)

