# python denoise_savitzky-golay.py -i diffusion/generated_gdstreamline_3.csv -o diffusion/generated_gdstreamline_3_denoised_3-21.csv -w 21 -p 3

import argparse
import logging
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("denoise_savgol")

def detect_spectral_columns(df):
    """
    Identifies columns that are likely spectral bands (numeric headers).
    Example: '400', '410.5', etc.
    """
    spectral_cols = []
    for col in df.columns:
        # Check if column name is a number (integer or float)
        if re.match(r"^\d+(\.\d+)?$", str(col)):
            spectral_cols.append(col)
    
    # Sort columns numerically to ensure the filter follows the spectrum correctly
    return sorted(spectral_cols, key=lambda x: float(x))

def denoise_spectra(df, window, poly):
    """
    Applies SG filter only to detected spectral columns, row by row.
    Preserves all other metadata columns exactly as they are.
    """
    cols = detect_spectral_columns(df)
    
    if not cols:
        logger.error("No spectral columns (numeric headers) detected!")
        return df

    logger.info(f"Found {len(cols)} spectral bands. Smoothing with window={window}, poly={poly}")
    
    # Work on a copy to prevent modifying original df in memory
    out_df = df.copy()
    
    # Extract spectral data
    # .fillna(0) is a safety measure; if a row is all NaN, the filter fails.
    # For a professional workflow, consider interpolating if only a few bands are missing.
    data = out_df[cols].values.astype(float)
    
    # Check for NaNs and warn
    if np.isnan(data).any():
        logger.warning("NaNs detected in spectral data. Filling with 0 for smoothing.")
        data = np.nan_to_num(data)

    # Apply filter across rows (axis=1)
    # Mode='interp' handles edges best for spectral data
    smoothed = savgol_filter(data, window_length=window, polyorder=poly, axis=1, mode="interp")
    
    # Put smoothed data back into the spectral columns
    out_df[cols] = smoothed
    
    return out_df

def main():
    parser = argparse.ArgumentParser(description="Patched SG Filter for ISPRS Spectral Data")
    parser.add_argument("-i", "--input", required=True, help="Input CSV")
    parser.add_argument("-o", "--output", required=True, help="Output CSV")
    parser.add_argument("-w", "--window", type=int, default=7, help="Window size (default: 7)")
    parser.add_argument("-p", "--poly", type=int, default=2, help="Polynomial order (default: 2)")
    
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Could not read file: {e}")
        return

    # Ensure window length is odd
    win = args.window
    if win % 2 == 0:
        win += 1
        logger.info(f"Adjusted window to {win} (must be odd)")

    # Run denoising
    df_denoised = denoise_spectra(df, win, args.poly)

    # Export
    df_denoised.to_csv(args.output, index=False)
    logger.info(f"Successfully saved denoised data to {args.output}")

if __name__ == "__main__":
    main()