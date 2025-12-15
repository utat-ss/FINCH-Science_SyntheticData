import wandb
import json
import torch
from torch.utils.data import DataLoader
from .data.data_preperation import get_unnormalizer

import logging
from .auxiliary import get_n_params, convert_tensors_to_ints
from .plotting import plot_to_wandb

def train_diffusion(cfg_train:(dict), cfg_export:(dict), diffusion_model:(torch.nn.Module), loss_fn:(torch.nn.Module), optimizer:(torch.optim), lr_scheduler:(torch.optim.lr_scheduler), configured_data):

    """
    This function is to train the conditional diffusion model
    """

    # Unpack the relevant dicts
    device = cfg_train['device']
    dtype = cfg_train['dtype']
    n_epoch = cfg_train['n_epoch'] # Number of epochs
    n_tb_epoch = cfg_train['n_tb_epoch'] # Number of training instances/batches per epoch (this means if have B as batch, total train samples will be n_tb * B)

    # Setup the save paths
    master_path = cfg_export['master_path']
    model_save = cfg_export['model_save']
    if not model_save.endswith('.pth'): model_save += '.pth'
    model_save = master_path + model_save
    test_save = cfg_export['test_save']
    if not test_save.endswith('.parquet'): test_save += '.parquet'
    test_save = master_path + test_save
    norm_save = cfg_export['norm_save']
    if not norm_save.endswith('.json'): norm_save += '.json'
    norm_save = master_path + norm_save
    locallog_save = master_path + cfg_export['locallog_save']

    # Get the iterators, the data unnormalizer etc.
    iterators, data_norm_dict = configured_data
    iter_train, iter_val, iter_test = iterators
    unnorm_lambda = get_unnormalizer(data_norm_dict)
    
    with open(norm_save, "w") as f: # Save the data_norm_dict so that we can unnormalize data in future synthesis cases
        json.dump(convert_tensors_to_ints(data_norm_dict), f, indent=4)

    diffusion_model.to(device) # Move the epsilon and all the vars to the device
    best_val_loss = float('inf') # We'll use this to save the best model

    # Log the param amount for model
    wandb.log({
        "model/param#": get_n_params(model=diffusion_model.epsilon)
    })
    logging.info(f"Training has started, will train for {n_epoch} epochs")
    for epoch in range(1, n_epoch + 1):

        ### TRAINING STEP
        # Set the model in training mode
        diffusion_model.train()
        total_train_loss = 0 # To accumulate the average loss

        logging.info(f"Epoch {epoch}, Training Step")
        for _ in range(n_tb_epoch):

            # Get the batch and unpack it, and move to the relevant device:
            batch = next(iter_train)

            x_0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            x_0 = x_0.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

            # Zero the grads
            optimizer.zero_grad()

            # Add noise, denoise, and get x_0 hat preds after internally augmenting the data (hat means pred)
            x_0_hat, x_0, x_n_hat, x_n = diffusion_model.training_procedure(x_0, abundances)
            # x_0 is augmented data, x_0_hat is the "fully recovered data", x_n is the added noise, x_n_hat is predicted noise

            # Calculate the loss
            train_loss, noise_loss, recons_loss = loss_fn(x_0_hat, x_0, x_n_hat, x_n)
            train_loss.backward() # Packprop the loss
            torch.nn.utils.clip_grad_norm_(diffusion_model.epsilon.parameters(), max_norm=1.0) # Clip grads
            optimizer.step()

            diffusion_model.update_ema() # Update the EMA parameters after each batch in training

            # Log train step results
            total_train_loss += train_loss.item()
            log_payload = {
                "train/train_loss_batch": train_loss.item(),
                "train/noise_loss_batch": noise_loss.item(),
                "epoch": epoch
            }
            if recons_loss is not None: log_payload['train/recons_loss_batch'] = recons_loss.item()
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

        ### VALIDATION STEP
        # Set model to eval
        diffusion_model.eval()

        logging.info(f"Epoch {epoch}, Validation Step")
        with torch.no_grad():

            batch = next(iter_val) # It is an infinite iterator

            # Get batch, seperate it, and move to the device
            x_0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            x_0 = x_0.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

            with diffusion_model.use_ema(): # Use the Exponential Moving Average model to sample
                x_0_hat, x_T = diffusion_model.sample(abundances) # Pass the abundances to get a prediction for our spectrum

            val_loss = loss_fn.sample_loss(x_0_hat, x_0)
            wandb.log({
                "general/validation_average_loss": val_loss.item()
            })
            logging.info(f"Epoch {epoch}, Average Val Loss: {val_loss.item()}")
            plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, epoch)

        # Save the diffusion model if it is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(diffusion_model.state_dict(), model_save)
            # Optional: Log best metric to summary
            wandb.run.summary["val/loss_best"] = best_val_loss
    
    logging.info(f"Finished {n_epoch} Epochs, Testing Step")

    ### TESTING STEP
    diffusion_model.eval()

    with torch.no_grad():

        batch = next(iter_test)

        # Get batch, seperate it, and move to the device
        x_0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
        x_0 = x_0.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

        with diffusion_model.use_ema(): # Use the Exponential Moving Average model to sample
            x_0_hat, x_T = diffusion_model.sample(abundances) # Pass the abundances to get a prediction for our spectrum

        test_loss = loss_fn.sample_loss(x_0_hat, x_0)

        wandb.log({
            "general/test_average_loss": test_loss.item()
        })
        plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, None) # Plot the testing results

    logging.info(f"Testing finished, Training of the Diffusion Model is Complete. The Difussion Model is saved at '{model_save}'. Now moving the model, logs, psi1/psi2, norm dict, etc. as artifacts to WandB.") 

    if cfg_export['save_to_wandb']:
        # Now dump every single thing to wandb
        # The trained diffusion model
        model_art = wandb.Artifact(name="trained_model", type="model")
        model_art.add_file(model_save)
        wandb.run.log_artifact(model_art)

        # Generated data through data initialization
        data_art = wandb.Artifact(name="run_data", type="dataset")
        data_art.add_file(cfg_export['psi1_path'])
        data_art.add_file(cfg_export['psi2_path'])
        wandb.run.log_artifact(data_art)

        # Data normalization specifics and test logs
        meta_art = wandb.Artifact(name="run_metadata", type="metadata")
        meta_art.add_file(locallog_save)
        meta_art.add_file(norm_save)
        wandb.run.log_artifact(meta_art)
