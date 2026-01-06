import ae

# from .epsilon import unet
# from .epsilon import mlp

import torch.optim as optim

import loss as custom_losses
import torch.nn as nn

from noise import noise_scheduling
# from .noise import noise_sampling
# from .data import data_augmentation

def load_vccae(cfg_ae_setup):

    ae_type = cfg_ae_setup['ae_type']
    cfg_ae = cfg_ae_setup['cfg_ae']

    if hasattr(ae, ae_type):
        ae_cls = getattr(ae, ae_type)
    else:
        raise ValueError(f"Unknown/Unsupported autoencoder type: {ae_type}")
    
    ae_model = ae_cls(**cfg_ae)
    return ae_model

# def load_epsilon(cfg_epsilon_setup):

#     epsilon_type = cfg_epsilon_setup['epsilon_type']
#     cfg_epsilon = cfg_epsilon_setup['cfg_epsilon']

#     if hasattr(unet, epsilon_type):
#         epsilon_cls = getattr(unet, epsilon_type)
#     elif hasattr(mlp, epsilon_type):
#         epsilon_cls = getattr(mlp, epsilon_type)
#     else:
#         raise ValueError(f"Unknown/Unsupported epsilon type: {epsilon_type}")

#     epsilon = epsilon_cls(**cfg_epsilon)
#     return epsilon

def load_optim(model, cfg_optim_setup): #epsilon excluded

    optim_type = cfg_optim_setup['optim_type']
    cfg_optim = cfg_optim_setup.get('cfg_optim', {})

    try:
        opt_cls = getattr(optim, optim_type)
    except:
        raise ValueError(f"Optimizer '{optim_type}' is not a valid optimizer in torch.optim. "
                         f"Check spelling (e.g., 'Adam' vs 'adam').")
    
    cfg_optim_temp = cfg_optim; cfg_optim_temp.pop('cfg_lrscheduler_setup', None)
    optimizer = opt_cls(model.parameters(), **cfg_optim_temp) #epsilon.parameters(), 

    lr_scheduler = None

    if 'cfg_lrscheduler_setup' in cfg_optim and cfg_optim['cfg_lrscheduler_setup']:

        lrscheduler_setup = cfg_optim['cfg_lrscheduler_setup']
        lrscheduler_type = lrscheduler_setup['lrscheduler_type']
        cfg_lrscheduler = lrscheduler_setup.get('cfg_lrscheduler', {})

        try:
            scheduler_cls = getattr(optim.lr_scheduler, lrscheduler_type)
        except:
            raise ValueError(f"Scheduler: {lrscheduler_type} does not exist in torch.optim.lr_scheduler")
        
        lr_scheduler = scheduler_cls(optimizer, **cfg_lrscheduler)

    return optimizer, lr_scheduler

def load_loss(cfg_loss_setup):

    loss_type = cfg_loss_setup['loss_type']
    cfg_loss = cfg_loss_setup.get('cfg_loss', {})

    if hasattr(custom_losses, loss_type):
        loss_cls = getattr(custom_losses, loss_type)
    elif hasattr(nn, loss_type):
        loss_cls = getattr(nn, loss_type)
    else:
        raise ValueError(f"Unknown/Unsupported loss in either loss.py or torch.nn: {loss_type}")

    loss = loss_cls(**cfg_loss)
    return loss

def load_scheduler(cfg_scheduler_setup):

    scheduler_type = cfg_scheduler_setup['scheduler_type']
    cfg_scheduler = cfg_scheduler_setup['cfg_scheduler']

    if hasattr(noise_scheduling, scheduler_type):
        scheduler_cls = getattr(noise_scheduling, scheduler_type)
    else:
        raise ValueError(f"Unknown/Unsupported noising scheduler type: {scheduler_type}")
    
    scheduler = scheduler_cls(**cfg_scheduler)
    return scheduler

# def load_tsampler(cfg_tsampler_setup):

#     tsampler_type = cfg_tsampler_setup['tsampler_type']
#     cfg_tsampler = cfg_tsampler_setup['cfg_tsampler']

#     if hasattr(noise_sampling, tsampler_type):
#         tsampler_cls = getattr(noise_sampling, tsampler_type)
#     else:
#         raise ValueError(f"Unknown/Unsupported time sampler type: {tsampler_type}")
    
#     tsampler = tsampler_cls(**cfg_tsampler)
#     return tsampler

# def load_augmenter(cfg_augmenter_setup):

#     augmenter_type = cfg_augmenter_setup['augmenter_type']
#     cfg_augmenter = cfg_augmenter_setup.get('cfg_augmenter', {})

#     if hasattr(data_augmentation, augmenter_type):
#         augmenter_cls = getattr(data_augmentation, augmenter_type)
#     else:
#         raise ValueError(f"Unknown/Unsupported augmenter type: {augmenter_type}")
    
#     augmenter = augmenter_cls(**cfg_augmenter)
#     return augmenter