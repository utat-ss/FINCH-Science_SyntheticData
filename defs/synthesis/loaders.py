"""
This file is to define all sorts of loaders from model loaders to abundance generators.
"""

from . import abundance_sampler
from .synth_func import *

import torch

def load_absampler(cfg_absampler_setup):
    absampler_type = cfg_absampler_setup['absampler_type']
    cfg_absampler = cfg_absampler_setup['cfg_absampler']

    if hasattr(abundance_sampler, absampler_type):
        absampler_cls = getattr(abundance_sampler, absampler_type)

    else:
        raise ValueError(f"Unknown/Unsupported abundance sampler of type: {absampler_type}")
    
    ab_sampler = absampler_cls(**cfg_absampler)

    return ab_sampler

def load_model(model_type:(str), model_statedict_path:(str), model_config_dict:(dict)):
    """
    The function to load models given some path to the model state dict.

    Args:
        model_type (str): The type of model we are using, either 'GaussianDiffusion' or 'AutoEncoder' for now
        model_statedict_path (str): The string of model's saved state dict's path
        model_config_dict (dict): The config dict for model we are using

    Returns:
        model (nn.Module): A fully loaded and initialized model, ready to synthesize
    """

    if model_type == 'GaussianDiffusion':
        model = _load_diffusion_model(model_config_dict, model_statedict_path)

    else:
        raise ValueError(f'Unknown/Unsupported synthesizer model type: {model_type}')
    
    return model

def _load_diffusion_model(cfg_diffusion_setup, cfg_diffusion_statedict_path):
    from ..diffusion.loaders import load_diffusion, load_epsilon, load_augmenter, load_scheduler, load_tsampler

    cfg_diffusion = cfg_diffusion_setup['cfg_diffusion']

    # Get the relevant dicts from cfg_diffusion, load epsilon, augmenter, scheduler, and t_sampler; put them into cfg_diff
    cfg_diffusion['epsilon'] = load_epsilon(cfg_diffusion['epsilon'])
    cfg_diffusion['augmenter'] = load_augmenter(cfg_diffusion['augmenter'])
    cfg_diffusion['scheduler'] = load_scheduler(cfg_diffusion['scheduler'])
    cfg_diffusion['t_sampler'] = load_tsampler(cfg_diffusion['t_sampler'])

    # Update the cfg_diffusion_setup with the newly updated cfg_diffusion
    cfg_diffusion_setup['cfg_diffusion'] = cfg_diffusion

    # Pass in cfg_diffusion_setup to the diffusion loader
    diffusion_model = load_diffusion(cfg_diffusion_setup)

    # Loads the learned parameters to the diffusion model
    diffusion_model.load_state_dict(torch.load(cfg_diffusion_statedict_path))

    return diffusion_model

def load_sampler(model_type:(str), model, ab_sampler):

    if model_type == 'GaussianDiffusion':

        spectra_sampler = GaussianDiffusionSampler(model, lean=False, ab_sampler=ab_sampler)

    elif model_type == 'AutoEncoder':
        
        spectra_sampler = AutoEncoderSampler(model, lean=False, ab_sampler=ab_sampler)

    else: 
        raise ValueError(f'Unknown/Unsupported synthesizer model type: {model_type}')
    
    return spectra_sampler


