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
from torch.utils.data import Dataset, DataLoader
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