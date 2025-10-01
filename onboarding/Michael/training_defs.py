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
    Want this function to read a csv file and return a torch tensor for analysis.
    """
    pass