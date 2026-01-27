"""
Search combinations of channel triples and n_former_blocks to find configs close to target parameter counts.
Prints results within +/-5% of targets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import yaml

from defs.diffusion.epsilon.unet import Epsilon_Cond1DUnet

def count_params(model):
    return sum(p.numel() for p in model.parameters())

candidates = [64, 128, 192, 256, 384, 512]
triples = [t for t in itertools.product(candidates, repeat=3) if t[0]<t[1]<t[2]]
# Keep only triples where all are divisible by dim_head=64
triples = [t for t in triples if all(x % 64 == 0 for x in t)]

targets = [25_000_000]
results = {t: [] for t in targets}

for channels in triples:
    for n_former in range(0,6):
        try:
            model = Epsilon_Cond1DUnet(
                n_bands=210,
                in_channel=1,
                out_channel=1,
                n_endmembers=3,
                channels=list(channels),
                n_former_blocks=n_former,
                n_mid_blocks=2,
                dropout=0.05,
                dim_head=64,
                down_type='conformer', mid_type='conformer', up_type='conformer',
                conv_kernel_size=3, conv_expansion_factor=2, time_embed_dim=64,
                wavembed_neighborhood=1000, wavembed_scale=1
            )
        except Exception as e:
            continue
        p = count_params(model)
        for target in targets:
            if abs(p - target) <= 0.05 * target:
                results[target].append((channels, n_former, p))

for target in targets:
    print(f"\nMatches for ~{target:,}:")
    for ch, nf, p in sorted(results[target], key=lambda x: abs(x[2]-target))[:10]:
        print(f"channels={ch}, n_former={nf} -> {p:,} params")

# If nothing found, print nearest candidates
if all(len(results[t])==0 for t in targets):
    print('\nNo close matches within 5%. Showing nearest candidates:')
    nearest = []
    for channels in triples:
        for n_former in range(0,6):
            try:
                model = Epsilon_Cond1DUnet(
                    n_bands=210,
                    in_channel=1,
                    out_channel=1,
                    n_endmembers=3,
                    channels=list(channels),
                    n_former_blocks=n_former,
                    n_mid_blocks=2,
                    dropout=0.05,
                    dim_head=64,
                    down_type='conformer', mid_type='conformer', up_type='conformer',
                    conv_kernel_size=3, conv_expansion_factor=2, time_embed_dim=64,
                    wavembed_neighborhood=1000, wavembed_scale=1
                )
            except Exception:
                continue
            p = count_params(model)
            nearest.append((channels, n_former, p))
    nearest.sort(key=lambda x: min(abs(x[2]-targets[0]), abs(x[2]-targets[1])))
    for ch, nf, p in nearest[:20]:
        print(f"channels={ch}, n_former={nf} -> {p:,} params")
