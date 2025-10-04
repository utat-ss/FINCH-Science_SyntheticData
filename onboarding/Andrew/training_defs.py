import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

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
