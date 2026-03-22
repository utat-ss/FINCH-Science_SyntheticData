("""Sanity-check: plot a few synthesized spectra.

Usage:
  python synthesized_sanitycheck.py [--csv PATH] [--n N] [--out PATH] [--no-show]

Loads the generated CSV in the same directory by default and plots a handful
of spectra (evenly sampled). Saves an output PNG by default.
""")

from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_wavelength_columns(df):
	# header contains numeric columns like '400','410',... detect those
	waves = [c for c in df.columns if str(c).strip().isdigit()]
	waves_sorted = sorted(waves, key=lambda c: int(c))
	return waves_sorted


def plot_spectra(csv_path: Path, n: int = 6, out: Path | None = None, show: bool = True):
	df = pd.read_csv(csv_path)
	wave_cols = find_wavelength_columns(df)
	if not wave_cols:
		raise RuntimeError('No wavelength columns found in CSV header')

	wavelengths = np.array([int(w) for w in wave_cols])

	# choose rows to plot: if fewer than n rows, plot all
	total = len(df)
	if total == 0:
		raise RuntimeError('CSV contains no spectra rows')

	if total <= n:
		idxs = list(range(total))
	else:
		# sample evenly plus include first/last
		step = max(1, total // n)
		idxs = list(range(0, total, step))[:n]

	plt.style.use('seaborn-v0_8')
	fig, ax = plt.subplots(figsize=(10, 5))

	for i in idxs:
		row = df.iloc[i]
		label = str(row.get('Spectra', f'row_{i}'))
		values = row[wave_cols].values.astype(float)
		ax.plot(wavelengths, values, label=label)

	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel('Reflectance / Value')
	ax.set_title(f'Sample of {len(idxs)} synthesized spectra from {csv_path.name}')
	ax.grid(True, linestyle=':', alpha=0.6)
	ax.legend(fontsize='small', ncol=2)

	if out is None:
		out = csv_path.with_name(csv_path.stem + '_spectra.png')
	fig.tight_layout()
	fig.savefig(out, dpi=200)
	if show:
		plt.show()
	print(f'Wrote plot to: {out}')


def main():
	p = argparse.ArgumentParser()
	default_csv = Path(__file__).with_name('generated_gdstreamline_3000.csv')
	p.add_argument('--csv', type=Path, default=default_csv)
	p.add_argument('--n', type=int, default=6, help='Number of spectra to plot')
	p.add_argument('--out', type=Path, default=None, help='Output PNG path')
	p.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
	args = p.parse_args()

	if not args.csv.exists():
		raise FileNotFoundError(f'CSV not found: {args.csv}')

	plot_spectra(args.csv, n=args.n, out=args.out, show=not args.no_show)


if __name__ == '__main__':
	main()

