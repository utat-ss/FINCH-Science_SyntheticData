#region Definitions

import os
import yaml
import argparse
import json

import torch

import subprocess

def cfg_from_args():
    """
    Function to get args from the .yaml args given in the script run code
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config.yaml file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    
    with open(args.config, 'r') as f:
        cfg_run = yaml.safe_load(f)

    return cfg_run

def parse_cfg_dict(cfg_run:(dict)):
    """
    Parser and reformer of the cfg dict.

    Args:
        cfg_run (dict): The cfg_run dict

    Returns:
        cfg_synthesis (dict): Details of synthesis, dict related to it
        cfg_export (dict): All the details related to exporting of synthesized data
    """

    # Get the main dicts
    cfg_synthesis = cfg_run['cfg_synthesis']
    cfg_export = cfg_run['cfg_export']
    cfg_absampler_setup = cfg_synthesis['cfg_absampler_setup']

    # Set the device
    device_str = cfg_synthesis['device']
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but is not available.")
    elif device_str != 'cpu':
        raise ValueError(f"Unknown/Unsupported device: {device_str}. Choose one of: 'cuda', 'cpu'")
    cfg_synthesis['device'] = torch.device(device_str)
    cfg_synthesis['cfg_absampler_setup']['cfg_absampler']['device'] = torch.device(device_str) # This was not passed in at the cfg yaml

    with open(cfg_synthesis['norm_dict_path']) as f:
        norm_dict = json.load(f)

    with open(cfg_synthesis['cfg_model_setup_path']) as f:
        cfg_model_setup = json.load(f)

    return cfg_synthesis, cfg_export, norm_dict, cfg_model_setup, cfg_absampler_setup

#endregion

#region Main Loop

from .loaders import *
from ..diffusion.data.data_preperation import get_unnormalizer
from ..diffusion.auxiliary import setup_logging, setup_wandb
from .synthesize import synthesize_data

import logging
import traceback

if __name__ == '__main__':

    # Get cfgs, parse 'em
    cfg_run = cfg_from_args()
    cfg_synthesis, cfg_export, norm_dict, cfg_model_setup, cfg_absampler_setup = parse_cfg_dict(cfg_run=cfg_run)

    # Setup log stuff
    run = setup_wandb(cfg_run, cfg_export)
    setup_logging(cfg_export)
    logging.info("Cfgs taken in, parsed. WandB and logging configured")

    # Initialize stuff
    unnormalizer = get_unnormalizer(norm_dict)
    model = load_model(cfg_synthesis['architecture'], cfg_synthesis['model_statedict_path'], cfg_model_setup).to(cfg_synthesis['device'])
    ab_sampler = load_absampler(cfg_absampler_setup)
    synthesizer = load_sampler(cfg_synthesis['architecture'], model, ab_sampler)
    logging.info("Unnormalizer and synthesizer model initialized")

    try:
        logging.info("Synthesis started")
        synthesize_data(cfg_synthesis, cfg_export, unnormalizer, synthesizer)
        logging.info("Synthesis successfully finished")

    except Exception as e:
        run.alert(title= "Synthesis crashed", text=str(e))
        traceback.print_exc()
        raise e
    
    finally:
        run.finish()

    print("Run complete. Shutting down VM!")
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

#endregion

