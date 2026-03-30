"""Run a non-interactive sanity-check to load models, synthesize two spectra, and save PNG+SVG.
This mirrors the notebook logic and prints exceptions for debugging.
"""
import sys
from pathlib import Path
import json
import traceback

# ensure repo root on path
repo_root = Path('c:/work/project/git-repo/FINCH-Science_SyntheticData').resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from defs.synthesis.loaders import load_model
from defs.synthesis.lean.lean_synth import get_lean_synthesis, synthesize
# unnormalizer utility
from defs.vae.data.data_preparation import get_unnormalizer


def find_wavelength_columns(df):
    waves = [c for c in df.columns if str(c).strip().isdigit()]
    return sorted(waves, key=lambda c: int(c))


def _resolve_artifact_paths(cfg_dict, cfg_key, folder_hint: Path):
    cfg_section = cfg_dict.get(cfg_key, {})
    data = cfg_section.get('data', {})
    if 'csv_path' in data and isinstance(data['csv_path'], str):
        p = Path(data['csv_path'])
        if not p.is_absolute():
            data['csv_path'] = str((folder_hint / data['csv_path']).resolve())
        cfg_section['data'] = data

    artifacts = cfg_section.get('artifacts', {})
    if artifacts:
        for k, v in list(artifacts.items()):
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    artifacts[k] = str((folder_hint / v).resolve())
        cfg_section['artifacts'] = artifacts
    cfg_dict[cfg_key] = cfg_section


def main():
    try:
        base = repo_root
        synth_dir = base / 'synthesis' / 'isprs'
        simpler_csv = base / 'data' / 'simpler_data_rwc.csv'
        psi_gd = synth_dir / 'diffusion' / 'psi2_gdstreamline.csv'
        psi_tcvae = synth_dir / 'tcvae' / 'psi2_tcvae.csv'

        print('Using base:', base)
        print('Reading ground-truth:', simpler_csv)
        gt = pd.read_csv(simpler_csv)
        pg = pd.read_csv(psi_gd)
        pt = pd.read_csv(psi_tcvae)

        gt_names = set(gt['Spectra'].astype(str))
        pg_names = set(pg['Spectra'].astype(str)) if 'Spectra' in pg.columns else set()
        pt_names = set(pt['Spectra'].astype(str)) if 'Spectra' in pt.columns else set()
        shared = list(gt_names & pg_names & pt_names)
        if len(shared) < 2:
            shared = list(gt['Spectra'].astype(str).iloc[:2])
        x_name, y_name = shared[0], shared[1]
        print('Selected spectra:', x_name, y_name)

        ab_cols = [c for c in gt.columns if c.lower().endswith('_fraction') or c in ['gv_fraction','npv_fraction','soil_fraction']]
        if not ab_cols:
            raise RuntimeError('Could not find abundance columns in ground-truth CSV')
        print('Abundance columns:', ab_cols)

        x_row = gt[gt['Spectra'].astype(str) == x_name].iloc[0]
        y_row = gt[gt['Spectra'].astype(str) == y_name].iloc[0]
        x_ab = x_row[ab_cols].values.astype(float)
        y_ab = y_row[ab_cols].values.astype(float)

        # cfg paths
        gd_cfg_path = synth_dir / 'diffusion' / 'cfg_diffusion_setup.json'
        gd_state_path = synth_dir / 'diffusion' / 'gdstreamline_statedict.pth'
        tcfg_path = synth_dir / 'tcvae' / 'cfg_tcvae_setup.json'
        tc_state_path = synth_dir / 'tcvae' / 'tcvae_statedict.pth'

        with open(gd_cfg_path, 'r') as fh:
            gd_cfg = json.load(fh)
        with open(tcfg_path, 'r') as fh:
            tc_cfg = json.load(fh)

        # resolve paths
        _resolve_artifact_paths(tc_cfg, 'cfg_tcvae', synth_dir / 'tcvae')
        _resolve_artifact_paths(gd_cfg, 'cfg_diffusion', synth_dir / 'diffusion')

        # ensure psi references point correctly
        try:
            tc_art = tc_cfg.get('cfg_tcvae', {}).get('artifacts', {})
            for key in ['psi1_path', 'psi2_path', 'psi1_normalized_path', 'psi2_normalized_path', 'psi1_spectra_path', 'psi2_spectra_path']:
                if key in tc_art:
                    p = Path(tc_art[key])
                    if not p.is_absolute():
                        tc_art[key] = str((synth_dir / 'diffusion' / p).resolve())
            tc_cfg['cfg_tcvae']['artifacts'] = tc_art
        except Exception:
            pass

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device:', device)

        # load models
        print('Loading GD model...')
        gd_model = load_model('GaussianDiffusion', str(gd_state_path), gd_cfg)
        print('Loading TC model...')
        tcvae_model = load_model('TCVAEncoder', str(tc_state_path), tc_cfg)

        # move models
        try:
            gd_model.to(device)
            tcvae_model.to(device)
        except Exception:
            pass

        print('Models loaded')

        # samplers and synthesize
        gd_sampler = get_lean_synthesis('GaussianDiffusion', gd_model)
        tc_sampler = get_lean_synthesis('TCVAEncoder', tcvae_model)
        cfg_lean = {'max_batch_size': 4, 'device': device}

        x_ab_t = torch.tensor(x_ab, dtype=torch.float32).unsqueeze(0)
        y_ab_t = torch.tensor(y_ab, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            x_gd_t = synthesize(cfg_lean, gd_sampler, x_ab_t)
            x_tc_t = synthesize(cfg_lean, tc_sampler, x_ab_t)
            y_gd_t = synthesize(cfg_lean, gd_sampler, y_ab_t)
            y_tc_t = synthesize(cfg_lean, tc_sampler, y_ab_t)

        x_gd = x_gd_t.cpu().numpy().squeeze()
        x_tc = x_tc_t.cpu().numpy().squeeze()
        y_gd = y_gd_t.cpu().numpy().squeeze()
        y_tc = y_tc_t.cpu().numpy().squeeze()

        # load norm dicts and get unnormalizers (prefer artifact norm_dict if present)
        gd_artifacts = gd_cfg.get('cfg_diffusion', {}).get('artifacts')
        if isinstance(gd_artifacts, dict):
            gd_norm_path = gd_artifacts.get('norm_dict_path')
            if gd_norm_path:
                gd_norm_path = Path(gd_norm_path)
                if not gd_norm_path.is_absolute():
                    gd_norm_path = (synth_dir / 'diffusion' / gd_norm_path).resolve()
        else:
            gd_norm_path = None
        if not gd_norm_path:
            gd_norm_path = synth_dir / 'diffusion' / 'norm_dict.json'

        tc_artifacts = tc_cfg.get('cfg_tcvae', {}).get('artifacts')
        if isinstance(tc_artifacts, dict):
            tc_norm_path = tc_artifacts.get('norm_dict_path')
            if tc_norm_path:
                tc_norm_path = Path(tc_norm_path)
                if not tc_norm_path.is_absolute():
                    tc_norm_path = (synth_dir / 'tcvae' / tc_norm_path).resolve()
        else:
            tc_norm_path = None
        if not tc_norm_path:
            tc_norm_path = synth_dir / 'tcvae' / 'norm_dict.json'

        # load JSONs and create unnormalizer functions
        try:
            with open(gd_norm_path, 'r') as fh:
                gd_norm = json.load(fh)
            unnorm_gd = get_unnormalizer(gd_norm)
        except Exception:
            unnorm_gd = lambda x: x

        try:
            with open(tc_norm_path, 'r') as fh:
                tc_norm = json.load(fh)
            unnorm_tc = get_unnormalizer(tc_norm)
        except Exception:
            unnorm_tc = lambda x: x

        # apply unnormalizers to synthesized spectra
        try:
            x_gd = unnorm_gd(x_gd)
            x_tc = unnorm_tc(x_tc)
            y_gd = unnorm_gd(y_gd)
            y_tc = unnorm_tc(y_tc)
        except Exception:
            # fall back silently if unnorm fails
            raise RuntimeError('Error applying unnormalizer to synthesized spectra')
            pass

        wave_cols = find_wavelength_columns(gt)
        wavelengths = np.array([int(w) for w in wave_cols])
        x_gt = gt[gt['Spectra'].astype(str) == x_name][wave_cols].values.astype(float).squeeze()
        y_gt = gt[gt['Spectra'].astype(str) == y_name][wave_cols].values.astype(float).squeeze()

        print('Synthesized shapes:', x_gd.shape, x_tc.shape)

        # plotting: enforce white background and visible gridlines
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10,5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.plot(wavelengths, x_gt, color='orange', linestyle='-', label=f'{x_name} (GT)')
        ax.plot(wavelengths, x_gd, color='orange', linestyle='--', label=f'{x_name} (GD)')
        ax.plot(wavelengths, x_tc, color='orange', linestyle=':', label=f'{x_name} (TCVAE)')
        ax.plot(wavelengths, y_gt, color='blue', linestyle='-', label=f'{y_name} (GT)')
        ax.plot(wavelengths, y_gd, color='blue', linestyle='--', label=f'{y_name} (GD)')
        ax.plot(wavelengths, y_tc, color='blue', linestyle=':', label=f'{y_name} (TCVAE)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
        ax.legend(fontsize='small', ncol=2)
        fig.tight_layout()

        out_png = synth_dir / 'sanitycompare_run.png'
        out_svg = synth_dir / 'sanitycompare_run.svg'
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_svg, dpi=200)
        print('Wrote', out_png, out_svg)

    except Exception as e:
        print('ERROR during run:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
