"""Append first N spectra from a generated CSV to the end of a psi1 CSV and write a new CSV.

Example:
  python hybridize_csv.py \
      --generated diffusion/generated_gdstreamline_3000.csv \
      --psi diffusion/psi1_gdstreamline.csv \
      --n 1000 \
      --output diffusion/psi1_gdstreamline_hybrid.csv

Fills missing columns with NaN, preserves original column order.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import pandas as pd


logger = logging.getLogger("hybridize_csv")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybridize psi1 and generated CSVs")
    p.add_argument("--generated", "-g", required=True, help="Path to generated CSV")
    p.add_argument("--psi", "-p", required=True, help="Path to psi1 CSV (base to append to)")
    p.add_argument("--n", "-n", type=int, default=1000, help="Number of rows from generated to append")
    p.add_argument("--output", "-o", default=None, help="Output CSV path (defaults to psi + _hybrid.csv)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def build_output_path(psi_path: str, out_arg: Optional[str]) -> str:
    if out_arg:
        return out_arg
    base, ext = os.path.splitext(psi_path)
    return f"{base}_hybrid{ext}"


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Reading psi CSV: %s", args.psi)
    df_psi = pd.read_csv(args.psi)
    logger.info("Reading generated CSV: %s", args.generated)
    df_gen = pd.read_csv(args.generated)

    n = args.n
    if n <= 0:
        logger.error("--n must be positive")
        return 2

    if len(df_gen) < n:
        logger.warning("Generated CSV has only %d rows; will append all of them", len(df_gen))
        n = len(df_gen)

    df_gen_head = df_gen.iloc[:n].copy()

    # Determine column order: preserve psi columns first, then any extra generated columns
    psi_cols = list(df_psi.columns)
    gen_cols = [c for c in df_gen_head.columns if c not in psi_cols]
    all_cols = psi_cols + gen_cols

    # Reindex dataframes to the union of columns (missing columns become NaN)
    df_psi_reidx = df_psi.reindex(columns=all_cols)
    df_gen_reidx = df_gen_head.reindex(columns=all_cols)

    df_out = pd.concat([df_psi_reidx, df_gen_reidx], ignore_index=True)

    out_path = build_output_path(args.psi, args.output)
    df_out.to_csv(out_path, index=False)
    logger.info("Wrote combined CSV to %s (psi rows: %d, appended rows: %d, total: %d)", out_path, len(df_psi), n, len(df_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
