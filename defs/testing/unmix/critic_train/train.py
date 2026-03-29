import torch
import torch.nn as nn
import torch.optim as optim

from ..auxiliary.auxiliary import get_n_params
from .auxiliary import get_r2, plot_abundances

import wandb
import logging
import pandas as pd

def train_critic(cfg_train:(dict), cfg_export:(dict), critic:(nn.Module), loss_fn:(nn.Module), optimizer:(optim), lr_scheduler:(optim.lr_scheduler), configured_data:(list)):

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
    iterators= configured_data # We do not unnormalize the input data, so we do not generate the unnorm lambda for that
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
            spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)
            # Ensure spectra have channel dimension: (batch, 1, B)
            if spectrum.ndim == 2:
                spectrum = spectrum.unsqueeze(1)

            # Zero the grads
            optimizer.zero_grad()

            # Get the predictions
            # If critic is an MLP (has Linear layers but no Conv1d), flatten spectrum
            has_linear = any(isinstance(m, torch.nn.Linear) for m in critic.modules())
            has_conv1d = any(isinstance(m, torch.nn.Conv1d) for m in critic.modules())
            inp = spectrum
            if inp.ndim == 3 and has_linear and not has_conv1d:
                inp = inp.view(inp.size(0), -1)

            pred_abundances = critic(inp)

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
            spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)
            # Ensure spectra have channel dimension: (batch, 1, B)
            if spectrum.ndim == 2:
                spectrum = spectrum.unsqueeze(1)

            # If critic is an MLP (has Linear layers but no Conv1d), flatten spectrum
            has_linear = any(isinstance(m, torch.nn.Linear) for m in critic.modules())
            has_conv1d = any(isinstance(m, torch.nn.Conv1d) for m in critic.modules())
            inp = spectrum
            if inp.ndim == 3 and has_linear and not has_conv1d:
                inp = inp.view(inp.size(0), -1)

            pred_abundances = critic(inp)
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
        spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)
        # Ensure spectra have channel dimension: (batch, 1, B)
        if spectrum.ndim == 2:
            spectrum = spectrum.unsqueeze(1)

        # If critic is an MLP (has Linear layers but no Conv1d), flatten spectrum
        has_linear = any(isinstance(m, torch.nn.Linear) for m in critic.modules())
        has_conv1d = any(isinstance(m, torch.nn.Conv1d) for m in critic.modules())
        inp = spectrum
        if inp.ndim == 3 and has_linear and not has_conv1d:
            inp = inp.view(inp.size(0), -1)

        pred_abundances = critic(inp)
        test_loss = loss_fn(pred_abundances, abundances)

        wandb.log({
            "general/test_average_loss": test_loss.item()
        })
        r2_payload = get_r2(ab_true=abundances, ab_pred=pred_abundances, mode='test')
        wandb.log(r2_payload)
        plot_abundances(ab_true=abundances, ab_pred=pred_abundances, mode='test')

        # Save predictions to parquet
        try:
            # sanitize name and orig_index to native Python types
            names_list = [
                (n.item().decode() if hasattr(n, 'item') and isinstance(n.item(), (bytes, bytearray)) else n.item())
                if hasattr(n, 'item') else str(n)
                for n in name
            ]
            orig_idx_list = [int(x.item()) if hasattr(x, 'item') else int(x) for x in orig_index]
            ab_true = abundances.cpu().numpy()
            ab_pred = pred_abundances.cpu().numpy()
            df_dict = {
                'name': names_list,
                'orig_index': orig_idx_list,
                'gv_true': ab_true[:, 0],
                'npv_true': ab_true[:, 1],
                'soil_true': ab_true[:, 2],
                'gv_pred': ab_pred[:, 0],
                'npv_pred': ab_pred[:, 1],
                'soil_pred': ab_pred[:, 2],
            }
            df = pd.DataFrame(df_dict)
            # write parquet to configured path
            df.to_parquet(test_save)
            logging.info(f"Saved test predictions to {test_save}")
        except Exception as e:
            logging.exception(f"Failed to save test predictions to {test_save}: {e}")

    logging.info(f"Testing finished, Training of the Diffusion Model is Complete. The Difussion Model is saved at '{model_save}'")

