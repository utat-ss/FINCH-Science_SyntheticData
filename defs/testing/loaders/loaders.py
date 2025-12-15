import torch.optim as optim
import torch.nn as nn
import torch

from ..critics.models import mlp
from ..critics.models import fno

from ..critic_train import custom_loss

def load_critic(cfg_critic_setup:(dict)):

    """
    Returns:
        critic (nn.Module): The critic of the generated data


    cfg_critic_setup:
        critic_type: type of the critic
        cfg_critic: the configs for the critic
    """

    critic_type = cfg_critic_setup['critic_type']
    cfg_critic = cfg_critic_setup['cfg_critic']

    if hasattr(mlp, critic_type):
        critic_cls = getattr(mlp, cfg_critic)
    elif hasattr(fno, critic_type):
        critic_cls = getattr(fno, critic_type)
    else:
        raise ValueError(f"Unknown/Unsupported critic type: {critic_type}")
    
    critic = critic_cls(**cfg_critic)

    return critic

def load_critictrain(cfg_critictrain_setup:(dict), critic:(nn.Module)):

    """
    Returns:
        optimizer (torch.optim): The optimizer
        lrscheduler (torch.optim.lrscheduler): The learning rate scheduler
        loss (torch.nn): The loss function


    cfg_critictrain:

        cfg_loss_setup:
            loss_type: type of loss
            cfg_loss: the configs for the loss

        cfg_optim_setup:
            optim_type: type of optimizer
            cfg_optim: the configs for the optimizer
            cfg_lrscheduler_setup:
                lrscheduler_type: type of lr scheduler
                cfg_lrscheduler: cfg of the lr scheduler
    """

    cfg_optim_setup = cfg_critictrain_setup['cfg_optim_setup']
    cfg_loss_setup = cfg_critictrain_setup['cfg_loss_setup']

    cfg_optim = cfg_optim_setup.get('cfg_optim', {})
    optim_type = cfg_optim_setup['optim_type']
    try:
        opt_cls = getattr(optim, optim_type)
    except:
        raise ValueError(f"Optimizer '{optim_type}' is not a valid optimizer in torch.optim. "
                         f"Check spelling (e.g., 'Adam' vs 'adam').")
    optimizer = opt_cls(critic.parameters(), **cfg_optim)

    lr_scheduler = None
    if 'cfg_lrscheduler_setup' in cfg_optim_setup and cfg_optim_setup['cfg_lrscheduler_setup']:
        lrscheduler_setup = cfg_optim_setup['cfg_lrscheduler_setup']
        lrscheduler_type = lrscheduler_setup['lrscheduler_type']
        cfg_lrscheduler = lrscheduler_setup.get('cfg_lrscheduler', {})
        try:
            scheduler_cls = getattr(optim.lr_scheduler, lrscheduler_type)
        except:
            raise ValueError(f"Scheduler: {lrscheduler_type} does not exist in torch.optim.lr_scheduler")
        lr_scheduler = scheduler_cls(optimizer, **cfg_lrscheduler)

    cfg_loss = cfg_loss_setup['cfg_loss']
    loss_type = cfg_loss_setup['loss_type']
    if hasattr(custom_loss, loss_type):
        loss_cls = getattr(custom_loss, loss_type)
    elif hasattr(nn, loss_type):
        loss_cls = getattr(nn, loss_type)
    else:
        raise ValueError(f"Unknown/Unsupported loss type: {loss_type}")
    loss = loss_cls(**cfg_loss)

    return optimizer, lr_scheduler, loss
