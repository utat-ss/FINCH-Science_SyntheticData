"""
This is the code to generate spectra leanly, i.e. without all the overhead of exporting the data etc, just synthesizing them
and returning the result.

One shall first use:
    get_lean_synthesis; to get the lean sampler (does not require absampler)

Then call:
    synthesize; given some abundance tensor, the synthesize all the spectra given abundances
"""

import os
import yaml
import argparse
import json

import torch
import torch.nn as nn

from ..synth_func import *

"""
get_lean_synthesis assumes the cfg_lean_synthesis as follows:

model_type (str): The type of model we are using, either 'GaussianDiffusion' or 'AutoEncoder' for now
max_batch_size (int): The maximum batch size used in sampling, this is to limit the load on gpu/cpu
"""

def get_lean_synthesis(cfg_lean_synthesis, model:(nn.Module)) -> SpectralSampler:

    """
    This function gets the lean synthesizer, it assumes the cfg_lean_synthesis has the keys states as above.

    Args:
        cfg_lean_synthesis (dict): The dict to get the lean synthesizer
    """

    if cfg_lean_synthesis['model_type'] == 'GaussianDiffusion':
        lean_sampler = GaussianDiffusionSampler(model=model, lean=True)
    elif cfg_lean_synthesis['model_type'] == 'AutoEncoder':
        lean_sampler = AutoEncoderSampler(model=model, lean=True)
    else:
        raise ValueError(f"Unknown/Unsupported model_type: {cfg_lean_synthesis['model_type']}")

    return lean_sampler

def synthesize(cfg_lean_synthesis, sampler:(SpectralSampler), ab_tensor:(torch.Tensor)) -> torch.Tensor:

    max_batch_size = cfg_lean_synthesis['max_batch_size']
    assert max_batch_size <= ab_tensor.shape[0], f"max_batch_size ({max_batch_size}) must be <= than batch of ab_tensor ({ab_tensor.shape[0]})" 
    
    if ab_tensor.device != next(sampler.model.parameters()).device: # Ensure the ab_tensor is in the same device as the sampler's model
        ab_tensor.to(next(sampler.model.parameters()).device)

    # We must parse the given ab_tensor into smaller parts, to ensure we don't have vram/ram space problems;
    split_ab_tensors = list(torch.split(ab_tensor, max_batch_size, dim=0))

    # Build the list of sampled spectra, turn it into a tensor at the end
    sampled_spectra = []
    for i in range(len(split_ab_tensors)):
        inter_tensor = split_ab_tensors.pop(0)
        sampled_spectra.append(sampler.predefined_ab_sample(inter_tensor))
    sampled_spectra = torch.tensor(sampled_spectra)

    return sampled_spectra
    

