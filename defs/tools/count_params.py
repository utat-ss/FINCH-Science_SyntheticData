"""
Instantiate the repo's Epsilon_Cond1DUnet with values from a config.yaml and print exact parameter count.
Usage:
  python tools/count_params_exact.py path/to/config.yaml

If required packages (e.g., conformer) are missing, the script will show the ImportError.
"""
import sys
import yaml
import argparse
from pathlib import Path

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('cfg')
    args = p.parse_args()

    cfg_path = Path(args.cfg)
    # Ensure FINCH-Science_SyntheticData is on sys.path so `import defs...` works
    root = Path(__file__).resolve().parents[1]  # FINCH-Science_SyntheticData
    sys.path.insert(0, str(root))

    with open(cfg_path, 'r', encoding='utf-8') as fh:
        doc = yaml.safe_load(fh)

    cfg_epsilon = doc['cfg_epsilon_setup']['cfg_epsilon']

    try:
        from defs.diffusion.epsilon.unet import Epsilon_Cond1DUnet
    except Exception as e:
        print('ERROR importing model:', e)
        raise

    # Map YAML keys to constructor
    kwargs = {
        'n_bands': cfg_epsilon.get('n_bands', 210),
        'in_channel': cfg_epsilon.get('in_channel', 1),
        'out_channel': cfg_epsilon.get('out_channel', 1),
        'n_endmembers': cfg_epsilon.get('n_endmembers', 3),
        'channels': cfg_epsilon.get('channels', [128,256,512]),
        'n_former_blocks': cfg_epsilon.get('n_former_blocks', 1),
        'n_mid_blocks': cfg_epsilon.get('n_mid_blocks', 1),
        'dropout': cfg_epsilon.get('dropout', 0.0),
        'dim_head': cfg_epsilon.get('dim_head', 64),
        'down_type': cfg_epsilon.get('down_type', 'conformer'),
        'mid_type': cfg_epsilon.get('mid_type', 'conformer'),
        'up_type': cfg_epsilon.get('up_type', 'conformer'),
        'conv_kernel_size': cfg_epsilon.get('conv_kernel_size', 3),
        'conv_expansion_factor': cfg_epsilon.get('conv_expansion_factor', 2),
        'time_embed_dim': cfg_epsilon.get('time_embed_dim', 64),
        'wavembed_neighborhood': cfg_epsilon.get('wavembed_neighborhood', 10000),
        'wavembed_scale': cfg_epsilon.get('wavembed_scale', 1),
    }

    print('Instantiating Epsilon_Cond1DUnet with:', kwargs)
    model = Epsilon_Cond1DUnet(**kwargs)
    print('Parameters (exact): {:,}'.format(count_params(model)))

if __name__ == '__main__':
    main()
