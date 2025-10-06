"""
Used guides:
    - Reading CSV files using pytorch: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

# Preamble
from sklearn.model_selection import train_test_split

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import transforms, utils


"""
NOTE: Will probably just use the "training" and "validation" records instead of splitting, actually.
"""

def load_csv(file):
    """
    This function reads a csv file.
    """

    data = pd.read_csv(file) # Read file using pandas
    return data

def test_train_split(data):
    """
    Split input data into testing and training tensors.

    NOTE: This function is engineered specifically for:
        - Inputs of 400-2490 nm wavelengths (10 nm intervals)
        - Outputs of gv_fraction, npv_fraction, soil_fraction (in that order)
    """
    train_input = []
    train_output = []
    test_input = []
    test_output = []

    start = 400
    end = 2490
    interval = 10
    frac = ['gv_fraction', 'npv_fraction', 'soil_fraction']

    for i in range(0, len(data['Spectra'])):
        # Split between training and testing
        if data['use'][i] == 'training':
            train_input.append([data[str(j)][i] for j in range(start, end + 1, interval)])
            train_output.append([data[j][i] for j in frac])
        else:
            test_input.append([data[str(j)][i] for j in range(start, end + 1, interval)])
            test_output.append([data[j][i] for j in frac])

    train_input = torch.tensor(train_input, dtype=torch.float64)
    train_output = torch.tensor(train_output, dtype=torch.float64)
    test_input = torch.tensor(test_input, dtype=torch.float64)
    test_output = torch.tensor(test_output, dtype=torch.float64)

    return train_input, train_output, test_input, test_output

# Adopted functions from Andrew's project

def load_spectral_data(filename="simpler_data_rwc.csv"):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)
    df = pd.read_csv(path)
    spectral_cols = [str(w) for w in range(400, 2491, 10)]
    X = df[spectral_cols].values.astype("float32")
    y = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
    return X, y

def get_dataloaders(X, y, batch_size=32, split_ratio=0.8):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    n_train = int(len(ds) * split_ratio)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True),
      DataLoader(val_ds, batch_size=batch_size)
    )