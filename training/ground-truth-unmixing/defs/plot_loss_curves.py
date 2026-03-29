import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

runs = [
    "fno_xs_seed42",
    "fno_s_seed42",
    "fno_m_seed42",
    "fno_l_seed42",
    "fno_xl_seed42",
]

for r in runs:
    run_dir = Path("runs") / r
    hist = run_dir / "history.csv"

    if not hist.exists():
        print("missing", hist)
        continue

    df = pd.read_csv(hist)

    # --- loss curve ---
    plt.figure()
    plt.plot(df["epoch"], df["train_mse"], label="train MSE")
    plt.plot(df["epoch"], df["val_mse"], label="val MSE")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE (log scale)")
    plt.title(r)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve_log.png", dpi=200)
    plt.close()

    # --- learning rate curve ---
    plt.figure()
    plt.plot(df["epoch"], df["lr"])
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("learning rate (log scale)")
    plt.title(r)
    plt.tight_layout()
    plt.savefig(run_dir / "lr_curve.png", dpi=200)
    plt.close()

    print("saved plots for", r)

print("done")

