#region Definitions

import os
import argparse
import yaml
import json

import torch

from .loaders.loaders import *
from .critic_train.train import train_critic
from .data.data import get_data
from ...diffusion.data.data_preperation import get_unnormalizer

import traceback
import logging

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
        cfg_import (dict): cfg dict related to imported data
        cfg_loader (dict): cfg dict related to preparing the loaders
        cfg_train (dict): cfg dict for the training of the critic model
        cfg_export (dict): cfg dict to export things to wandb
        cfg_critic_setup (dict): cfg dict to load the critic
        cfg_critictrain_setup (dict): cfg dict to handle the optimizer, lr_scheduler, and loss functions of the critic
    """
    # Config parsing
    cfg_import = cfg_run['cfg_import']
    cfg_loader = cfg_run['cfg_loader']
    cfg_train = cfg_run['cfg_train']
    cfg_export = cfg_run['cfg_export']

    cfg_critic_setup = cfg_run['cfg_critic_setup']
    cfg_critictrain_setup = cfg_run['cfg_critictrain_setup']

    # Dtype config
    dtype_str = cfg_train['dtype']
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
        'float64': torch.float64
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown/Unsupported dtype: {dtype_str}. Choose one of: 'float16', 'bfloat16', 'float32', 'float64'")
    cfg_train['dtype'] = dtype_map[dtype_str]

    # Device config
    device_str = cfg_train['device']
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but is not available.")
    elif device_str != 'cpu':
        raise ValueError(f"Unknown/Unsupported device: {device_str}. Choose one of: 'cuda', 'cpu'")
    cfg_train['device'] = torch.device(device_str)

    with open(cfg_import['norm_dict_path']) as f:
        norm_dict = json.load(f)
    cfg_import['unnormalizer'] = get_unnormalizer(norm_dict)
    cfg_import.pop('norm_dict_path', None)

    return cfg_import, cfg_loader, cfg_train, cfg_export, cfg_critic_setup, cfg_critictrain_setup

def unmix_script(cfg_export_master:(dict), cfg_unmix:(dict)):

    """
    Is the main 'script' used to get the unmix metrics, this is a functionalized and regularly updated version of code in this file.

    Args:
        cfg_export_master (dict): Export configs obtained from the master cfg, this is different than the one in cfg_unmix
        cfg_unmix (dict): Cfgs to run this specific metric, for details, check example_cfg.yaml in testing/master/

    Returns:
        Logs the training metrics
    """

    # Parse the cfg_unmix dict
    cfg_import, cfg_loader, cfg_train, cfg_export, cfg_critic_setup, cfg_critictrain_setup = parse_cfg_dict(cfg_unmix)
    logging.info("Cfgs taken in, parsed")

    # Sort out the data
    configured_data = get_data(cfg_import, cfg_loader)
    logging.info("Data configured")

    # Load the critic, optimizer, lr_scheduler, loss_fn
    critic = load_critic(cfg_critic_setup)
    optimizer, lr_scheduler, loss_fn = load_critictrain(cfg_critictrain_setup, critic)
    logging.info("Critic, optimizer, lr_scheduler, loss_fn configured")

    logging.info("Training function called")
    train_critic(cfg_train, cfg_export, critic, loss_fn, optimizer, lr_scheduler, configured_data)
    logging.info("Training finished, synthesized data tested")

#endregion
