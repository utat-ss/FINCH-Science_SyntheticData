import torch
import torch.nn as nn
import torch.optim as optim

from ..auxiliary.auxiliary import get_n_params
from .auxiliary import get_r2, plot_abundances

import wandb
import logging

def train_critic(cfg_train:(dict), cfg_export:(dict), critic:(nn.Module), loss_fn:(nn.Module), optimizer:(optim), lr_scheduler:(optim.lr_scheduler), configured_data):

    """
    This is the function which trains the critic model on ksi_train (synthesized from psi_1), then validates and tests on ksi_val and ksi_test (parts of psi_2).

    Args:
        cfg_train (dict): The config for the training run
        cfg_export (dict): The config for the exports of results
        critic (nn.Module): The critic model to be trained
        loss_fn (nn.Module): The loss function to be used during training
        optimizer (optim): Optimizer to update the critic during training
        lr_scheduler (optim.lr_scheduler): Learning scheduler to change the learning rate during training
        configured_data (list[Iterator]): A list of the configured iterators [ksi_train, ksi_val, ksi_test]

    Returns:
        ...
    """

    # Get the keys from the training dict
    device = cfg_train['device']
    dtype = cfg_train['dtype']
    n_epoch = cfg_train['n_epoch']
    n_tb_epoch = cfg_train['n_tb_epoch']

    # Unpack the configured data list, get iterators and unnorming function
    iterators, data_norm_dict = configured_data # We do not unnormalize the input data, so we do not generate the unnorm lambda for that
    iter_train, iter_val, iter_test = iterators

    # Throw the critic to device
    critic.to(device)

    # Setup save paths
    model_save = cfg_export['model_save']
    if not model_save.endswith('.pth'): model_save += '.pth'
    test_save = cfg_export['test_save']
    if not test_save.endswith('.parquet'): test_save += '.parquet'
    best_val_loss = float('inf') # We'll use this to save the best model

    # Log the param amount of the critic
    wandb.log({
        "critic/param#": get_n_params(model=critic)
    })
    logging.info(f"Training has started, will train for {n_epoch} epochs")
    for epoch in range(1, n_epoch + 1):

        ### TRAINING STEP ###
        # Set the critic in training mode
        critic.train()
        total_train_loss = 0.0

        logging.info(f"Epoch {epoch}, Training Step")
        for _ in range(n_tb_epoch):

            # Get the batch and unpack it
            batch = next(iter_train)
            spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']

            # Zero the grads
            optimizer.zero_grad()

            # Get the predictions
            pred_abundances = critic(spectrum)

            # Calculate the loss, backprop it, and take optimizer step
            train_loss = loss_fn(pred_abundances, abundances)
            train_loss.backward()
            optimizer.step()

            # Log train step results
            total_train_loss += train_loss.item()
            log_payload = {
                "train/train_loss_batch": train_loss.item(),
                "epoch": epoch
            }
            wandb.log(log_payload)

        # Log average train results
        wandb.log({
            "general/train_average_loss": total_train_loss/(n_tb_epoch)
        })
        logging.info(f"Epoch {epoch}, Average Train Loss: {total_train_loss/(n_tb_epoch)}")

        # Step Scheduler, if exists
        if lr_scheduler is not None:
            lr_scheduler.step()
            logging.info("LR Scheduler updated")

        ### VALIDATION STEP ###
        # Set critic to evaluation
        critic.eval()
        logging.info(f"Epoch {epoch}, Validation Step")

        with torch.no_grad():
            # Get the batch and unpack it
            batch = next(iter_val)
            spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']

            pred_abundances = critic(spectrum)
            val_loss = loss_fn(pred_abundances, abundances)

            wandb.log({
                "general/validation_average_loss": val_loss.item()
            })
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss.item()}")
            r2_payload = get_r2(ab_true=abundances, ab_pred=pred_abundances, mode='val')
            wandb.log(r2_payload)

        # Save the critic if it is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(critic.state_dict(), model_save)
            # Optional: Log best metric to summary
            wandb.run.summary["val/loss_best"] = best_val_loss

    logging.info(f"Finished {n_epoch} Epochs, Testing Step")
    ### TESTING STEP ###
    critic.eval()

    with torch.no_grad():

        # Get the batch and unpack it
        batch = next(iter_test)
        spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']

        pred_abundances = critic(spectrum)
        test_loss = loss_fn(pred_abundances, abundances)

        wandb.log({
            "general/test_average_loss": test_loss.item()
        })
        r2_payload = get_r2(ab_true=abundances, ab_pred=pred_abundances, mode='test')
        wandb.log(r2_payload)
        plot_abundances(ab_true=abundances, ab_pred=pred_abundances, mode='test')

    logging.info(f"Testing finished, Training of the Diffusion Model is Complete. The Difussion Model is saved at '{model_save}'")

