#region Definitions

import os
import argparse
import yaml

import torch

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
        cfg_data (dict): cfg dict to prepare the real and synthesized datasets
        cfg_train (dict): cfg dict for the training of the critic model
        cfg_export (dict): cfg dict to export things to wandb

        cfg_critic_setup (dict): cfg dict to load the critic
        cfg_critictrain_setup (dict): cfg dict to handle the optimizer, lr_scheduler, and loss functions of the critic
    """
    # Config parsing
    cfg_data = cfg_run['cfg_data']
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

    return cfg_data, cfg_train, cfg_export, cfg_critic_setup, cfg_critictrain_setup

#endregion

#region Main Loop

from .loaders.loaders import *
from .critic_train.train import train
from .auxiliary.auxiliary import *
from .data.data import get_data

import traceback
import logging

if __name__ == "__main__":

    # Get the configs
    cfg_run = cfg_from_args()
    cfg_data, cfg_train, cfg_export, cfg_critic_setup, cfg_critictrain_setup = parse_cfg_dict(cfg_run)

    # Setup wandb and logging
    run = setup_wandb(cfg_run, cfg_export)
    setup_logging(cfg_export)

    logging.info("Cfgs taken in, parsed. WandB and logging configured")

    # Figure out data
    configured_data = get_data(cfg_data)

    logging.info("Data has been parsed and configured")

    # Load the critic, optimizer, lr_scheduler, loss_fn
    critic = load_critic(cfg_critic_setup)
    optimizer, lr_scheduler, loss_fn = load_critictrain(cfg_critictrain_setup, critic)

    logging.info("Critic, optimizer, lrscheduler, and loss has been configured")

    try:
        logging.info("Training function called")
        train(cfg_train, cfg_export, critic, loss_fn, optimizer, lr_scheduler, configured_data)
        logging.info("Training finished, synthesized data tested")

    except Exception as e:
        run.alert(title= "Training crashed", text=str(e))
        traceback.print_exc()
        raise e
    
    finally:
        run.finish()

#endregion