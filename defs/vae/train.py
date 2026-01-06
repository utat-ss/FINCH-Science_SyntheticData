import torch
import wandb
import json
import torch
from torch.utils.data import DataLoader
from training_defs import *

# Not needed yet (Also directly copied from diffusion)
# from .auxiliary import get_n_params, convert_tensors_to_ints
# from .plotting import plot_to_wandb

import logging

def train_vccae(cfg_train:(dict),
                cfg_export:(dict),
                vccae_model:(torch.nn.Module),
                loss_fn:(torch.nn.Module),
                optimizer:(torch.optim),
                lr_scheduler:(torch.optim.lr_scheduler),
                configured_data
                ):
    """
    Training the variational conditional convolutional autoencoder
    Thanks Ege for helping me out

    Note: Batch size must be set to 1 for this to work
    """

    device = cfg_train['device']
    dtype = cfg_train['dtype']
    n_epoch = cfg_train['n_epoch']

    #Save paths
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

    # with open(norm_save, "w") as f: # Save the data_norm_dict so that we can unnormalize data in future synthesis cases
    #     json.dump(convert_tensors_to_ints(data_norm_dict), f, indent=4)

    # Model setup
    vccae_model.to(device)
    best_val_loss = float('inf')

    # Data setup
    train_loader, val_loader = configured_data

    """
    # Log the param amount for model
    wandb.log({
        "model/param#": get_n_params(model=diffusion_model.epsilon)
    })
    """
    logging.info(f'Training start. {n_epoch} epochs.')
    for epoch in range(n_epoch):

        # Training
        vccae_model.train()
        tot_train_loss = 0
        logging.info(f'Epoch {epoch + 1}, Training step')

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            estimated, mu, variation = vccae_model(xb, yb)
            optimizer.zero_grad()
            train_loss = loss_fn(estimated, xb, mu, variation)
            train_loss.backward()
            optimizer.step()

            tot_train_loss += float(train_loss)
            log_payload = {
                'train_loss': float(train_loss),
                'epoch': epoch + 1
            }
            # wandb.log(log_payload)
        
        # Log average train results
        # wandb.log({
        #     "general/train_average_loss": tot_train_loss)
        # })
        logging.info(f"Epoch {epoch}, Average Train Loss: {tot_train_loss}")

        # Step Scheduler, if exists
        if lr_scheduler is not None:
            lr_scheduler.step()
            logging.info("LR Scheduler updated")

        ### VALIDATION STEP
        # Set model to eval
        vccae_model.eval()

        logging.info(f"Epoch {epoch}, Validation Step")
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device) # spectrum
                yb = yb.to(device) # abundance
                estimated, mu, variation = vccae_model(xb, yb)
                tot_val += loss_fn(estimated, xb, mu, variation) * xb.size(0)
            
            # wandb.log({
            #     "general/validation_average_loss": tot_val
            # })
            logging.info(f"Epoch {epoch}, Average Val Loss: {tot_val}")
            # Dunno this yet
            # plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, epoch)

        # Save the model if it is the best
        if tot_val < best_val_loss:
            best_val_loss = tot_val
            torch.save(vccae_model.state_dict(), model_save)
            # Optional: Log best metric to summary
            # wandb.run.summary["val/loss_best"] = best_val_loss

    logging.info(f"Finished {n_epoch} Epochs, Testing Step")

    ### TESTING STEP
    vccae_model.eval()

    with torch.no_grad():
        for xb, yb in val_loader:
                xb = xb.to(device) # spectrum
                yb = yb.to(device) # abundance
                estimated, mu, variation = vccae_model(xb, yb)
                test_loss += loss_fn(estimated, xb, mu, variation) * xb.size(0)

        # wandb.log({
        #     "general/test_average_loss": test_loss.item()
        # })

        # plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, None) # Plot the testing results

    logging.info(f"Testing finished, Training complete. Model saved at '{model_save}'.") 


    # Export as artifacts to wandb?
    # if cfg_export['save_to_wandb']:
    #     # Now dump every single thing to wandb
    #     # The trained model
    #     model_art = wandb.Artifact(name="trained_model", type="model")
    #     model_art.add_file(model_save)
    #     wandb.run.log_artifact(model_art)

    #     # Generated data through data initialization
    #     data_art = wandb.Artifact(name="run_data", type="dataset")
    #     data_art.add_file(cfg_export['psi1_path'])
    #     data_art.add_file(cfg_export['psi2_path'])
    #     wandb.run.log_artifact(data_art)

    #     # Data normalization specifics and test logs
    #     meta_art = wandb.Artifact(name="run_metadata", type="metadata")
    #     meta_art.add_file(locallog_save)
    #     meta_art.add_file(norm_save)
    #     wandb.run.log_artifact(meta_art)