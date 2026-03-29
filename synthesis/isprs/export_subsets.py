"""
export_subsets.py

Read a CSV into a pandas DataFrame, export the first N rows for configured
cutoffs (default 3000 and 9000) and the full file as separate CSVs.

Usage examples:
    python export_subsets.py generated_gdstreamline.csv
    python export_subsets.py generated_gdstreamline.csv --out-prefix subset
    python export_subsets.py generated_gdstreamline.csv --rows 3000 9000 15000
"""

from pathlib import Path
import argparse
import sys
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export CSV subsets: first N rows and all rows")
    p.add_argument("input",
                   help="Input CSV file path")
    p.add_argument("--out-prefix",
                   help="Output filename prefix (default: input filename without extension)")
    p.add_argument("--rows",
                   nargs="+",
                   type=int,
                   default=[3000, 9000],
                   help="Row cutoffs to export (default: 3000 9000)")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    
    if not inp.exists():
        print(f"Error: input file not found: {inp}", file=sys.stderr)
        return 1

    out_prefix = args.out_prefix or inp.stem

    # Read the full dataframe
    try:
        df = pd.read_csv(inp)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    total_rows = len(df)
    # Clean and sort cutoffs, removing duplicates and those larger than the dataset
    cutoffs = sorted({n for n in args.rows if 0 < n < total_rows})

    for n in cutoffs:
        out_path = inp.parent / f"{out_prefix}_{n}.csv"
        df.head(n).to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({n} rows)")

    # Export full file
    full_out = inp.parent / f"{out_prefix}_all.csv"
    df.to_csv(full_out, index=False)
    print(f"Wrote {full_out} ({total_rows} rows)")

    return 0

if __name__ == "__main__":
    sys.exit(main())