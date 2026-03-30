"""Compute FFTs for spectra and compare original vs denoised (SG) versions.

Usage examples:
  python fft_compare.py \
    --orig tcvae/generated_tcvae_3000.csv \
    --denoised tcvae/generated_tcvae_3000_denoised.csv \
    --select TCVAE_0,TCVAE_1,TCVAE_2,TCVAE_3,TCVAE_4,TCVAE_5,TCVAE_6,TCVAE_7 \
    --outdir ./fft_output

The script will produce one CSV per selected spectrum containing
frequency, original FFT (real/imag), denoised FFT (real/imag), and difference.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_wavelength_columns(df):
    """Identifies only numeric column headers (e.g., '400', '410')."""
    return [c for c in df.columns if re.fullmatch(r"\d+", str(c))]

def compute_fft(y, wavelengths):
    n = y.size
    delta = np.mean(np.diff(wavelengths))
    freqs = np.fft.rfftfreq(n, d=delta)
    fft_val = np.fft.rfft(y)
    return freqs, fft_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", default="tcvae/generated_tcvae_3000.csv")
    parser.add_argument("--denoised", default="tcvae/generated_tcvae_3000_denoised.csv")
    parser.add_argument("--select", help="Comma-separated names, e.g., TCVAE_0,TCVAE_1")
    parser.add_argument("--outdir", default="fft_results")
    args = parser.parse_args()

    # Load Data
    df_orig = pd.read_csv(args.orig)
    df_den  = pd.read_csv(args.denoised)
    
    # Identify numeric columns
    wcols = detect_wavelength_columns(df_orig)
    wavelengths = np.array([float(x) for x in wcols])
    os.makedirs(args.outdir, exist_ok=True)

    # Filter selection
    if args.select:
        target_names = [s.strip() for s in args.select.split(",")]
        df_orig = df_orig[df_orig.iloc[:, 0].astype(str).isin(target_names)]

    for _, row in df_orig.iterrows():
        name = str(row.iloc[0])
        
        # Find matching row in denoised file
        den_row = df_den[df_den.iloc[:, 0].astype(str) == name]
        if den_row.empty:
            print(f"Skipping {name}: No matching row in denoised file.")
            continue
            
        print(f"Processing {name}...")
        y_orig = row[wcols].values.astype(float)
        y_den  = den_row[wcols].values.astype(float).flatten()

        # Fourier Transforms
        freqs, fft_orig = compute_fft(y_orig, wavelengths)
        _, fft_den = compute_fft(y_den, wavelengths)

        # Complex Subtraction (Requirement: Subtract the two FTs)
        fft_diff = fft_orig - fft_den

        # Save results
        res_df = pd.DataFrame({
            "frequency": freqs,
            "diff_mag": np.abs(fft_diff),
            "diff_real": fft_diff.real,
            "diff_imag": fft_diff.imag
        })
        res_df.to_csv(os.path.join(args.outdir, f"{name}_fft_diff.csv"), index=False)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, np.abs(fft_orig), label="Original FT", alpha=0.7)
        plt.semilogy(freqs, np.abs(fft_den), label="Denoised (SG) FT", alpha=0.7)
        plt.semilogy(freqs, np.abs(fft_diff), label="Difference (Noise Removed)", linestyle="--")
        plt.title(f"Fourier Domain Analysis: {name}")
        plt.xlabel("Frequency (1/nm)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        svg_path = os.path.join(args.outdir, f"{name}_plot.svg")
        png_path = os.path.join(args.outdir, f"{name}_plot.png")
        plt.savefig(svg_path)
        plt.savefig(png_path)
        plt.close()

if __name__ == "__main__":
    main()