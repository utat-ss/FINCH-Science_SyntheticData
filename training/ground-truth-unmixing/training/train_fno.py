from __future__ import annotations

import os
import time
import json
from typing import Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from defs.training_defs import create_dataloader, fit_train_scaler
from models.fno_unmixing import FNO1DUnmixing

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

print("train_fno.py started")
def load_cfg(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if path.endswith((".yml", ".yaml")):
        import yaml
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            raise ValueError(f"config file is empty: {path}")
        return cfg
    raise ValueError("config must be .yaml/.yml or .json")


@torch.no_grad()
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0, keepdim=True)) ** 2, dim=0)
    r2 = 1.0 - (ss_res / (ss_tot + eps))
    return float(torch.mean(r2).item())


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: Optional[float] = None) -> float:
    model.train()
    total = 0.0
    n = 0

    for X, y in loader:
        # X: (batch, B) -> (batch, 1, B)
        X = X.unsqueeze(1).to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X)  # (batch, K)
        loss = criterion(pred, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = X.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0

    ys, ps = [], []
    for X, y in loader:
        X = X.unsqueeze(1).to(device)
        y = y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        bs = X.size(0)
        total += float(loss.item()) * bs
        n += bs

        ys.append(y.detach().cpu())
        ps.append(pred.detach().cpu())

    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(ps, dim=0)

    return {"mse": total / max(n, 1), "r2": r2_score(y_true, y_pred)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    run_root = cfg["training"].get("run_dir", "runs")
    run_name = cfg["training"].get("run_name", "unnamed_run")
    run_dir = os.path.join(run_root, run_name)
    ensure_dir(run_dir)

    # save config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,train_mse,val_mse,val_r2,lr,time_sec\n")


    # ---- data ----
    df = pd.read_csv(cfg["data"]["csv_path"])
    seed = int(cfg.get("seed", 42))

    val_frac = float(cfg["data"].get("val_frac", 0.15))
    test_frac = float(cfg["data"].get("test_frac", 0.15))
    target_cols = cfg["data"].get("target_cols", ["gv_fraction", "npv_fraction", "soil_fraction"])
    K = len(target_cols)

    train_df, temp_df = train_test_split(df, test_size=(val_frac + test_frac), random_state=seed, shuffle=True)
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=seed, shuffle=True)

    # IMPORTANT: fit ONE scaler on TRAIN only
    scaler = fit_train_scaler(train_df)

    batch_size = int(cfg["training"].get("batch_size", 64))
    train_loader = create_dataloader(train_df, batch_size=batch_size, shuffle=True, scaler=scaler, scale_data=True, target_cols=target_cols)
    val_loader   = create_dataloader(val_df,   batch_size=batch_size, shuffle=False, scaler=scaler, scale_data=True, target_cols=target_cols)
    test_loader  = create_dataloader(test_df,  batch_size=batch_size, shuffle=False, scaler=scaler, scale_data=True, target_cols=target_cols)

    # ---- model ----
    mcfg = cfg["model"]
    model = FNO1DUnmixing(
        num_endmembers=K,
        modes=int(mcfg["modes"]),
        width=int(mcfg["width"]),
        num_layers=int(mcfg["num_layers"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        pool=str(mcfg.get("pool", "mean")),
    )

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("use_cuda", True)) else "cpu")
    model = model.to(device)

    # ---- train ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    scheduler = None
    if bool(cfg["training"].get("use_plateau_scheduler", True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg["training"].get("plateau_factor", 0.5)),
            patience=int(cfg["training"].get("plateau_patience", 7)),
        )

    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["patience"])
    grad_clip = cfg["training"].get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    ckpt_name = cfg["training"].get("checkpoint_name", "fno_best.pt")
    ckpt_path = os.path.join(run_dir, "best.pt")


    early = EarlyStopping(patience=patience)
    best_val = float("inf")

    print(f"device={device}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  B={train_loader.dataset.X.shape[1]}  K={K}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_mse = val_metrics["mse"]

        if scheduler is not None:
            scheduler.step(val_mse)

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"[{epoch:04d}/{epochs}] "
            f"train_mse={train_mse:.6e}  val_mse={val_mse:.6e}  val_r2={val_metrics['r2']:.4f}  "
            f"lr={lr:.3e}  time={dt:.1f}s"
        )

        with open(history_path, "a") as f:
            f.write(f"{epoch},{train_mse:.8e},{val_mse:.8e},{val_metrics['r2']:.6f},{lr:.6e},{dt:.2f}\n")


        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(ckpt_path, {"model_state": model.state_dict(), "best_val_mse": best_val, "cfg": cfg})

        if early.step(val_mse):
            print(f"early stopping. best_val_mse={best_val:.6e}")
            break

    # ---- test ----
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"[TEST] mse={test_metrics['mse']:.6e}  r2={test_metrics['r2']:.4f}")
    print(f"best checkpoint: {ckpt_path}")
    
    # ---- write summary (paper-friendly) ----
    summary = {
        "run_name": run_name,
        "best_val_mse": best_val,
        "test_mse": float(test_metrics["mse"]),
        "test_r2": float(test_metrics["r2"]),
        "epochs_ran": int(epoch),
        "device": str(device),
        "model": cfg["model"],
        "training": {k: cfg["training"].get(k) for k in ["batch_size","learning_rate","epochs","patience","grad_clip"]},
        "data": cfg["data"],
        "checkpoint_path": ckpt_path,
        "history_path": history_path,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
