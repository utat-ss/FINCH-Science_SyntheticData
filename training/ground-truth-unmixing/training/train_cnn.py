import os
import json
import time
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    import yaml
except ImportError:
    yaml = None

from models.cnn_unmixing import CNNUnmixing1D


# --- optional reuse of your existing scaling/dataset utilities ---
try:
    from defs.training_defs import fit_train_scaler as repo_fit_train_scaler  # type: ignore
except Exception:
    repo_fit_train_scaler = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise RuntimeError("scikit-learn is required (StandardScaler).") from e


TARGET_DEFAULT = ["gv_fraction", "npv_fraction", "soil_fraction"]


class SpectraDataset(Dataset):
    """
    fallback dataset if you don't want to import the repo one.
    expects df with feature_cols and target_cols present.
    returns:
      X: (B,)
      y: (K,)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]):
        self.X = df[feature_cols].to_numpy(dtype=np.float32)
        self.y = df[target_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".yaml", ".yml"]:
            if yaml is None:
                raise RuntimeError("pyyaml not installed, but yaml config provided.")
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError("config must be .yaml/.yml or .json")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


import re

def infer_band_feature_cols(df: pd.DataFrame, target_cols: List[str], num_bands: int) -> List[str]:
    """
    Pick exactly `num_bands` feature columns (spectral bands).

    Strategy:
    1) take numeric columns excluding targets
    2) try to interpret each column name as a wavelength (e.g. "400", "410", "band_400", "wl_410", "b420")
    3) sort by wavelength and take first `num_bands`

    If we can't detect wavelength-like names, we fallback to the first `num_bands` numeric columns.
    """
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in set(target_cols)
    ]

    def to_wavelength(col_name: str):
        s = str(col_name)

        # case 1: column name is literally a number like "400" or "410"
        try:
            v = float(s)
            if 300 <= v <= 2600:
                return v
        except Exception:
            pass

        # case 2: column name contains digits like "band_400", "wl_410", "b420"
        m = re.search(r"(\d{3,4})", s)
        if m:
            v = float(m.group(1))
            if 300 <= v <= 2600:
                return v

        return None

    wl_cols = []
    other_cols = []
    for c in numeric_cols:
        wl = to_wavelength(str(c))
        if wl is None:
            other_cols.append(c)
        else:
            wl_cols.append((wl, c))

    # if we found wavelength-like cols, use them
    if len(wl_cols) >= num_bands:
        wl_cols.sort(key=lambda x: x[0])
        chosen = [c for _, c in wl_cols[:num_bands]]
        return chosen

    # fallback: just take the first num_bands numeric cols
    return numeric_cols[:num_bands]

def split_df(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1"
    n = len(df)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_train = int(math.floor(train_frac * n))
    n_val = int(math.floor(val_frac * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def fit_train_scaler(train_df: pd.DataFrame, feature_cols: List[str]):
    # use repo helper if present (your “fit ONE scaler on train only” rule)
    if repo_fit_train_scaler is not None:
        try:
            # supports either (train_df) or (train_df, feature_cols)
            try:
                return repo_fit_train_scaler(train_df, feature_cols)  # type: ignore
            except TypeError:
                return repo_fit_train_scaler(train_df)  # type: ignore
        except Exception:
            pass

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: List[str], scaler) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    out = df.copy()
    out.loc[:, feature_cols] = Xs
    return out


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    n = 0

    ys = []
    yhs = []

    for X, y in loader:
        X = X.to(device)               # (batch, B)
        y = y.to(device)               # (batch, K)
        X = X.unsqueeze(1)             # (batch, 1, B)
        yhat = model(X)                # (batch, K)

        mse = torch.mean((yhat - y) ** 2).item()
        bs = X.shape[0]
        mse_sum += mse * bs
        n += bs

        ys.append(y.detach().cpu())
        yhs.append(yhat.detach().cpu())

    y_all = torch.cat(ys, dim=0).numpy()
    yh_all = torch.cat(yhs, dim=0).numpy()

    # multi-output R2 averaged over outputs
    sse = np.sum((y_all - yh_all) ** 2, axis=0)
    sst = np.sum((y_all - np.mean(y_all, axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    r2_per = 1.0 - (sse / sst)
    r2 = float(np.mean(r2_per))

    return {
        "mse": float(mse_sum / max(n, 1)),
        "r2": r2,
    }


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, opt: torch.optim.Optimizer, grad_clip: Optional[float]) -> float:
    model.train()
    mse_sum = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)       # (batch, B)
        y = y.to(device)       # (batch, K)
        X = X.unsqueeze(1)     # (batch, 1, B)

        yhat = model(X)
        loss = torch.mean((yhat - y) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()

        bs = X.shape[0]
        mse_sum += float(loss.item()) * bs
        n += bs

    return float(mse_sum / max(n, 1))


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    run_name = cfg.get("run", {}).get("name", "cnn_run")
    seed = int(cfg.get("run", {}).get("seed", 42))
    out_root = cfg.get("run", {}).get("output_dir", "runs")
    run_dir = os.path.join(out_root, f"{run_name}_{now_stamp()}_seed{seed}")
    ensure_dir(run_dir)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ---
    data_cfg = cfg.get("data", {})
    csv_path = data_cfg["csv_path"]
    target_cols = data_cfg.get("target_cols", TARGET_DEFAULT)

    df = pd.read_csv(csv_path)

    model_cfg = cfg.get("model", {})
    num_bands_expected = int(model_cfg.get("num_bands", 210))

    feature_cols = infer_band_feature_cols(df, target_cols, num_bands=num_bands_expected)

    if len(feature_cols) != num_bands_expected:
        raise ValueError(
            f"expected {num_bands_expected} band columns, but selected {len(feature_cols)}. "
            "check your CSV column names / band detection."
        )

    print(f"[data] selected {len(feature_cols)} feature cols (bands). "
        f"first={feature_cols[:3]} last={feature_cols[-3:]}")


    split_cfg = data_cfg.get("split", {})
    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.15))
    test_frac = float(split_cfg.get("test", 0.15))
    shuffle = bool(data_cfg.get("shuffle", True))

    train_df, val_df, test_df = split_df(df, train_frac, val_frac, test_frac, seed=seed, shuffle=shuffle)

    scaler = fit_train_scaler(train_df, feature_cols)
    train_df = apply_scaler(train_df, feature_cols, scaler)
    val_df = apply_scaler(val_df, feature_cols, scaler)
    test_df = apply_scaler(test_df, feature_cols, scaler)

    batch_size = int(cfg.get("train", {}).get("batch_size", 1024))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_loader = DataLoader(SpectraDataset(train_df, feature_cols, target_cols), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(SpectraDataset(val_df, feature_cols, target_cols), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(SpectraDataset(test_df, feature_cols, target_cols), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    print(
        f"[dl] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"batch={batch_size} workers={num_workers}",
        flush=True,
)


    # --- model ---
    model_cfg = cfg.get("model", {})
    num_bands = int(model_cfg.get("num_bands", len(feature_cols)))
    num_endmembers = int(model_cfg.get("num_endmembers", len(target_cols)))

    model = CNNUnmixing1D(
        num_bands=num_bands,
        num_endmembers=num_endmembers,
        channels=model_cfg.get("channels", [64, 128, 256]),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        norm=model_cfg.get("norm", "batch"),
        residual=bool(model_cfg.get("residual", True)),
        res_blocks_per_stage=int(model_cfg.get("res_blocks_per_stage", 1)),
        pool=model_cfg.get("pool", "avg"),
    ).to(device)

    # --- train setup ---
    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    max_epochs = int(train_cfg.get("max_epochs", 200))
    grad_clip = train_cfg.get("grad_clip_norm", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    early_patience = int(train_cfg.get("patience", 20))
    min_delta = float(train_cfg.get("min_delta", 0.0))

    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 5)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6))
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history: List[Dict[str, Any]] = []

    ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
    history_path = os.path.join(run_dir, "history.csv")
    summary_path = os.path.join(run_dir, "summary.json")


    print(f"[train] starting training on device={device} | "
      f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} | "
      f"batch_size={batch_size} | max_epochs={max_epochs}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        # 1) train
        train_mse = train_epoch(model, train_loader, device, opt, grad_clip)

        # 2) validate
        val_metrics = eval_epoch(model, val_loader, device)
        val_mse = val_metrics["mse"]
        val_r2 = val_metrics["r2"]

        # 3) scheduler step (based on val loss)
        scheduler.step(val_mse)

        # 4) read current lr AFTER scheduler step
        cur_lr = float(opt.param_groups[0]["lr"])

        # 5) optional: print lr reductions (compare to previous epoch lr)
        if len(history) > 0:
            prev_lr = float(history[-1]["lr"])
            if cur_lr < prev_lr:
                print(f"[lr] reduced: {prev_lr:.3e} -> {cur_lr:.3e} at epoch {epoch}", flush=True)
        epoch_time = time.time() - epoch_start

        # 6) progress print
        print(
            f"[epoch {epoch:03d}] {epoch_time:.2f}s lr={cur_lr:.3e} train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_r2={val_r2:.4f}",
            flush=True,
        )

        # 7) log row
        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "epoch_time_s": epoch_time
        }
        history.append(row)
        avg_time = sum(h.get("epoch_time_s", 0.0) for h in history) / len(history)
        print(f"         avg_epoch={avg_time:.2f}s", flush=True)

        pd.DataFrame(history).to_csv(history_path, index=False)

        # 8) checkpoint + early stopping
        improved = (best_val - val_mse) > min_delta
        if improved:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "config": cfg,
                    "feature_cols": feature_cols,
                    "target_cols": target_cols,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= early_patience:
            break


    # --- test with best checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_best = eval_epoch(model, train_loader, device)
    val_best = eval_epoch(model, val_loader, device)
    test_best = eval_epoch(model, test_loader, device)

    # generalization gap at best epoch (val - train)
    gap = float(val_best["mse"] - train_best["mse"])

    summary = {
        "run_name": run_name,
        "seed": seed,
        "device": str(device),
        "run_dir": run_dir,
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "train_mse_at_best": float(train_best["mse"]),
        "val_mse_at_best": float(val_best["mse"]),
        "test_mse": float(test_best["mse"]),
        "val_r2": float(val_best["r2"]),
        "test_r2": float(test_best["r2"]),
        "generalization_gap": gap,
        "epochs_ran": int(len(history)),
        "num_features": int(len(feature_cols)),
        "num_targets": int(len(target_cols)),
        "config": cfg,
    }
    save_json(summary_path, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
