import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SpectraDataset(Dataset):
    def __init__(self, df=None, csv_path=None, scale_data=True):
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        assert df is not None, "You must provide either a DataFrame or csv_path"

        spectral_cols = [str(wl) for wl in range(400, 2500, 10)]
        X = df[spectral_cols].values
        y = df[['gv_fraction', 'npv_fraction', 'soil_fraction']].values

        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloader(df, batch_size=32, shuffle=True, scale_data=True):
    dataset = SpectraDataset(df, scale_data=scale_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_loss_optimizer(model, lr=1e-3):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return loss_fn, optimizer
