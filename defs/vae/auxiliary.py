def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def setup_wandb(cfg_run, cfg_export):
    import wandb

    settings = wandb.Settings(
        show_errors=True,
        silent=False,
        show_warnings=True
    )

    run = wandb.init(
        entity= cfg_export['entity'],
        project= cfg_export['project'],
        name= cfg_export.get('name', 'default'),
        config= cfg_run,
        job_type= 'training',
        settings=settings
    )

    return run

def setup_logging(cfg_export):
    import logging
    import sys

    master_path = cfg_export['master_path']
    locallog_save = cfg_export['local_log'] #locallog_save
    if not locallog_save.endswith('.txt'): locallog_save += '.txt'
    locallog_path = master_path + locallog_save

    logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(locallog_path, mode='w'), # 'w' overwrites, 'a' appends
        logging.StreamHandler(sys.stdout)                  # Keeps console output active for W&B
        ]
    )

def convert_tensors_to_ints(data):
    """
    Recursively converts all single-element PyTorch tensors in a dictionary 
    to standard Python integers.
    """
    import torch

    if isinstance(data, dict):
        return {k: convert_tensors_to_ints(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [convert_tensors_to_ints(i) for i in data]
    
    elif isinstance(data, torch.Tensor):
        # Check if the tensor is a single entry (scalar)
        if data.numel() == 1:
            return float(data.item())
        else:
            # Return original tensor if it contains multiple elements
            return data
            
    else:
        return data
    
def save_diffusion_inits(cfg_export, cfg_diffusion_setup, cfg_epsilon_setup, cfg_augmenter_setup, cfg_scheduler_setup, cfg_tsampler_setup):
    import json

    modelinits_save = cfg_export['modelinits_save']
    master_path = cfg_export['master_path']
    if not modelinits_save.endswith('.json'): modelinits_save += '.json'
    modelinits_save = master_path + modelinits_save

    cfg_diffusion_addon = {
        'epsilon': cfg_epsilon_setup,
        'augmenter': cfg_augmenter_setup,
        'scheduler': cfg_scheduler_setup,
        't_sampler': cfg_tsampler_setup
    }
    cfg_diffusion_setup['cfg_diffusion'].update(cfg_diffusion_addon)

    with open(modelinits_save, 'w') as f:
        json.dump(cfg_diffusion_setup, f, indent=4)