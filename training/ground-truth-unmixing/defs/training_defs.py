import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_spectral_cols():
    # 400..2490 step 10 -> 210 bands
    return [str(wl) for wl in range(400, 2500, 10)]


def fit_train_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """
    Fit ONE scaler on TRAIN spectra only.
    Reuse this scaler for val/test. No exceptions.
    """
    spectral_cols = get_spectral_cols()
    X_train = train_df[spectral_cols].values
    scaler = StandardScaler().fit(X_train)
    return scaler


class SpectraDataset(Dataset):
    """
    Returns:
      X: (B,) float32
      y: (K,) float32
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        csv_path: str | None = None,
        scaler: StandardScaler | None = None,
        scale_data: bool = True,
        target_cols: list[str] | None = None,
    ):
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        assert df is not None, "provide df or csv_path"

        if target_cols is None:
            target_cols = ["gv_fraction", "npv_fraction", "soil_fraction"]

        spectral_cols = get_spectral_cols()
        X = df[spectral_cols].values
        y = df[target_cols].values

        if scale_data:
            if scaler is None:
                raise ValueError(
                    "scale_data=True but scaler=None. "
                    "Fit ONE scaler on TRAIN and reuse it for val/test."
                )
            X = scaler.transform(X)

        self.X = torch.from_numpy(X).float()  # (N, B)
        self.y = torch.from_numpy(y).float()  # (N, K)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    scaler: StandardScaler | None = None,
    scale_data: bool = True,
    target_cols: list[str] | None = None,
):
    dataset = SpectraDataset(df=df, scaler=scaler, scale_data=scale_data, target_cols=target_cols)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loss_optimizer(model, lr=1e-3):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return loss_fn, optimizer
