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
        cfg_data (dict): cfg dict to prepare the datasets
        cfg_train (dict): cfg dict for the training of the diffusion model
        cfg_export (dict): cfg dict to export things to wandb

        cfg_epsilon_setup (dict): cfg dict for the epsilon model to be used in diffusion
        cfg_augmenter_setup (dict): cfg dict for the augmenter to be used in the diffusion model
        cfg_scheduler_setup (dict): cfg dict for the noise scheduler
        cfg_tsampler_setup (dict): cfg dict for the timestep sampler


        cfg_optim_setup (dict): cfg dict for the optimizer to be used in training
        cfg_loss_setup (dict): cfg dict for the loss function to be used during training
    """
    # Config parsing
    cfg_data = cfg_run['cfg_data']
    cfg_train = cfg_run['cfg_train']
    cfg_export = cfg_run['cfg_export']

    cfg_diffusion_setup = cfg_run['cfg_diffusion_setup']
    cfg_epsilon_setup = cfg_run['cfg_epsilon_setup']
    cfg_augmenter_setup = cfg_run['cfg_augmenter_setup']
    cfg_scheduler_setup = cfg_run['cfg_scheduler_setup']
    cfg_tsampler_setup = cfg_run['cfg_tsampler_setup']

    cfg_optim_setup = cfg_run['cfg_optim_setup']
    cfg_loss_setup = cfg_run['cfg_loss_setup']

    # Add psi paths to the export path
    cfg_export['psi1_path'] = cfg_data['cfg_dataset_save']['psi1_path']
    cfg_export['psi2_path'] = cfg_data['cfg_dataset_save']['psi2_path']

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

    return cfg_data, cfg_train, cfg_export, cfg_diffusion_setup, cfg_epsilon_setup, cfg_augmenter_setup, cfg_scheduler_setup, cfg_tsampler_setup, cfg_optim_setup, cfg_loss_setup

#endregion

#region Main Loop

from .loaders import *
from .data.data_preperation import *
from .train import train_diffusion
from .auxiliary import setup_wandb, setup_logging

import traceback
import logging

if __name__ == "__main__":

    # Get the configs
    cfg_run = cfg_from_args()
    cfg_data, cfg_train, cfg_export, cfg_diffusion_setup, cfg_epsilon_setup, cfg_augmenter_setup, cfg_scheduler_setup, cfg_tsampler_setup, cfg_optim_setup, cfg_loss_setup = parse_cfg_dict(cfg_run)

    # Setup wandb and logging
    run = setup_wandb(cfg_run, cfg_export)
    setup_logging(cfg_export)

    logging.info("Cfgs taken in, parsed. WandB and logging configured")

    # Figure out data
    configured_data = get_data(cfg_data)

    logging.info("Data has been parsed and configured")

    # Initialize stuff
    cfg_diffusion_addon = {
        'epsilon': load_epsilon(cfg_epsilon_setup),
        'augmenter': load_augmenter(cfg_augmenter_setup),
        'scheduler': load_scheduler(cfg_scheduler_setup),
        't_sampler': load_tsampler(cfg_tsampler_setup)
    }
    cfg_diffusion_setup['cfg_diffusion'].update(cfg_diffusion_addon)
    diffusion_model = load_diffusion(cfg_diffusion_setup)

    optimizer, lr_scheduler = load_optim(cfg_optim_setup, diffusion_model.epsilon)
    loss_fn = load_loss(cfg_loss_setup)

    logging.info("The diffusion model, optimizer, loss function etc. has been configured")

    try:
        logging.info("Training function started")
        train_diffusion(cfg_train, cfg_export, diffusion_model, loss_fn, optimizer, lr_scheduler, configured_data)
        logging.info("Training finished, everything logged to wandb.")

    except Exception as e:
        run.alert(title= "Training crashed", text=str(e))
        traceback.print_exc()
        raise e

    finally:
        run.finish()
    
#endregion