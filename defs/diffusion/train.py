import wandb
import json
import torch
from torch.utils.data import DataLoader
from .data.data_preperation import get_unnormalizer
import gc

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
    verbose = cfg_train['verbose']
    use_autocast = cfg_train.get('use_autocast', False)

    # Enable autocast if asked:
    scaler = torch.amp.GradScaler(device=device, enabled=use_autocast)

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
    loss_str_tracker = {
        'train': 'Train',
        'noise': 'Noise',
        'recons': 'Recons',
        'fft': 'FFT',
        'tv': 'TV'
    }

    # Log the param amount for model
    wandb.log({
        "model/param#": get_n_params(model=diffusion_model.epsilon)
    })
    logging.info(f"Training has started, will train for {n_epoch} epochs")
    for epoch in range(1, n_epoch + 1):

        ### TRAINING STEP
        # Set the model in training mode
        diffusion_model.train()
        total_losses = {k: 0.0 for k in loss_str_tracker.keys()} # Initialize the total losses tracker dict

        logging.info(f"Epoch {epoch}: Training Step{f', LR: {lr_scheduler.get_last_lr()[0]:.8f}' if lr_scheduler else ''}")
        for _ in range(n_tb_epoch):

            # Get the batch and unpack it, and move to the relevant device:
            batch = next(iter_train)

            x_0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            x_0 = x_0.to(device=device, dtype=dtype, non_blocking=True); abundances = abundances.to(device=device, dtype=dtype, non_blocking=True)

            # Zero the grads
            optimizer.zero_grad()

            # Apply everything if use_autocast=True with AMP(float16 base), if =False, just defaults to the classic forward and backward passes without AMP
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast): # This uses the autocast (automatic mixed precision, AMP) if enabled 
                # Add noise, denoise, and get x_0 hat preds after internally augmenting the data (hat means pred)
                x_0_hat, x_0, x_n_hat, x_n, t = diffusion_model.training_procedure(x_0, abundances)
                # x_0 is augmented data, x_0_hat is the "fully recovered data", x_n is the added noise, x_n_hat is predicted noise
                # Calculate the loss
                train_loss, noise_loss, recons_loss, fft_loss, tv_loss = loss_fn(x_0_hat, x_0, x_n_hat, x_n, unnorm_lambda)
            scaler.scale(train_loss).backward() # If AMP is enabled, scale the loss first and then backprop it
            scaler.unscale_(optimizer) # Unscale the grads of optimizer's assigned params, in-place, if AMP is enabled, we have to manually call this because we want to clip grads
            # Use configurable grad clip norm from cfg_train (default 1.0)
            max_norm = cfg_train.get('grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(diffusion_model.epsilon.parameters(), max_norm=max_norm) # Clip grads
            scaler.step(optimizer) # Takes an optimizer step if inf/NaNs are not present
            scaler.update() # Updates the scale factor

            diffusion_model.update_ema() # Update the EMA parameters after each batch in training

            # Log train step results
            current_step = [
                ('train', train_loss),
                ('noise', noise_loss),
                ('recons', recons_loss),
                ('fft', fft_loss),
                ('tv', tv_loss)
            ] # Log all the losses
            log_payload = {"epoch": epoch}; print_parts = [] # Init the log payload
            for key, tensor in current_step: 
                if tensor is not None:
                    val = tensor.item() # Gets the singular item
                    total_losses[key] += val # Accumulates it to total
                    log_payload[f'train/{key}_loss_batch'] = val # Adds to the WandB payload
                    print_parts.append(f'{loss_str_tracker[key]}_Loss: {val:.4f}') # Adds to the print string
            if verbose:
                logging.info(f'T Tensor: {t.tolist()}')
                logging.info(f'Pred Var: {x_n_hat.var().item():.4f} | GT Var: {x_n.var().item():.4f}')
                logging.info(' | '.join(print_parts))
            wandb.log(log_payload)

        # Log average train results
        log_payload, print_parts = {}, []
        for key, display_name in loss_str_tracker.items():
            total = total_losses.get(key, 0)
            if total != 0:
                avg = total / n_tb_epoch
                if key == 'train':
                    wb_key = "general/train_average_loss"
                else:
                    wb_key = f"general_detailed/train_average_{key}_loss"
                log_payload[wb_key] = avg
                print_parts.append(f"Average {display_name} Loss: {avg:.4f}")
        wandb.log(log_payload)
        if print_parts:
            print_parts[0] = f"Epoch {epoch} Train Losses: " + print_parts[0]
            logging.info(' | '.join(print_parts))

        # Step Scheduler, if exists
        if lr_scheduler is not None:
            lr_scheduler.step()
            logging.info("LR Scheduler updated")

        ### VALIDATION STEP
        # Set model to eval
        diffusion_model.eval()

        # Release all the cache from training
        gc.collect()
        torch.cuda.empty_cache()

        logging.info(f"Epoch {epoch}: Validation Step")
        with torch.no_grad():

            batch = next(iter_val) # It is an infinite iterator

            # Get batch, seperate it, and move to the device
            x_0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            x_0 = x_0.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

            with diffusion_model.use_ema(): # Use the Exponential Moving Average model to sample
                x_0_hat, x_T = diffusion_model.sample(abundances) # Pass the abundances to get a prediction for our spectrum

            v_total, v_recons, v_fft, v_tv = loss_fn.sample_loss(x_0_hat, x_0, unnorm_lambda)
            val_metrics = [
                (v_total,  'Total', 'general/validation_average_loss'),
                (v_recons, 'Recons',   'general_detailed/validation_average_recons_loss'),
                (v_fft,    'FFT',      'general_detailed/validation_average_fft_loss'),
                (v_tv,     'TV',       'general_detailed/validation_average_tv_loss')
            ]
            log_payload, print_parts = {}, []
            for tensor, display_name, key in val_metrics:
                if tensor is not None and tensor.item() != 0:
                    val = tensor.item()
                    log_payload[key] = val
                    print_parts.append(f"{display_name} Loss: {val:.4f}")
            wandb.log(log_payload)
            if print_parts:
                print_parts[0] = f"Epoch {epoch} Val Losses: " + print_parts[0]
                logging.info(' | '.join(print_parts))
            plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, epoch)

        # Delete used stuff
        del x_0, x_0_hat, x_T, abundances, batch
        gc.collect()
        torch.cuda.empty_cache()

        # Save the diffusion model if it is the best
        if v_total < best_val_loss:
            best_val_loss = v_total
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

        t_total, t_recons, t_fft, t_tv = loss_fn.sample_loss(x_0_hat, x_0, unnorm_lambda)
        t_metrics = [
            (t_total, 'Test Loss', 'general/test_average_loss'),
            (t_recons, 'Recons', 'general_detailed/test_average_recons_loss'),
            (t_fft, 'FFT', 'general_detailed/test_average_fft_loss'),
            (t_tv, 'TV', 'general_detailed/test_average_tv_loss')
        ]
        log_payload, print_parts = {}, []
        for tensor, display_name, key in t_metrics:
            if tensor is not None and tensor.item() != 0:
                val = tensor.item()
                log_payload[key] = val
                print_parts.append(f"{display_name} Loss: {val:.4f}")
        wandb.log(log_payload)
        if print_parts:
            print_parts[0] = "Test Losses: " + print_parts[0]
            logging.info(' | '.join(print_parts))

        plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 20, None) # Plot the testing results

    logging.info(f"Testing finished, Training of the Diffusion Model is Complete. The Difussion Model is saved at '{model_save}'. Now moving the model, logs, psi1/psi2, norm dict, etc. as artifacts to WandB if enabled.") 

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
