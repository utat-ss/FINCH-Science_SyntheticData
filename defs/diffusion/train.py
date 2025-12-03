import wandb
import torch
from torch.utils.data import DataLoader
from .data.data_preperation import get_unnormalizer

from sklearn.metrics import r2_score, mean_absolute_error
import logging
from .auxiliary import get_n_params
from .plotting import plot_to_wandb


def train_diffusion_deprecated(cfg_train: dict, cond_diffusion, loss, optimizer: torch.optim, dataloaders: list[DataLoader]):

    """
    The function to train and validate the models

    Args:
        cfg_train (dict): 
            'cfg_loader' (dict):
                - 'test' (int): How many samples to be used to test at the end
                - 't_batch' (int): How many train batches in total, to be trained on
                - 'validate' (int): How many samples to validate on each epoch
                - 'epoch' (int): How many epochs in total
            - 'range' (list): Spectral range [lower, upper]
            - 'device' : The device being used
            - 'tot_samples' (int): Total samples in the dataset
        cfg_diffusion (dict):
            't_sampler' (Sampling): A temperature sampling object
        cond_diffusion (class): The conditional diffusion class, already initialized
        loss (class): Loss function class, must have MSE and SAM at the very least
        optimizer (torch.optim): The optimizer to use
        dataloaders (list[DataLoader]): A list of dataloaders, in order: [train, validate, test]
    """

    # Dataload the giant dataset into train, test, validation datasets
    cfg_loader = cfg_train.get(
        'cfg_loader', {'test': 23, 't_batch': 1, 'validate': 4, 'epoch': 50}
    ) # Get the configs for the loader first

    dl_train, dl_val, dl_test = dataloaders[0], dataloaders[1], dataloaders[2]

    
    iterator_train = iter(dl_train); iterator_validation = iter(dl_val); iterator_test = iter(dl_test)

    n_test = cfg_loader.get('test', 23)
    n_epoch = cfg_loader.get('epoch', 50)
    n_t_batch = cfg_loader.get('t_batch', 1)
    n_validate = cfg_loader.get('validate', 4)

    # Infer the amount of n_train, ensure completeness
    n_train = int(cfg_train['tot_samples'] - n_epoch*n_validate - n_test)
    n_train_per_epoch = n_train // n_epoch


    ### TRAINING, VALIDATION, TEST LOOP ###

    # Define the init dictionary where we will keep track of the generated spectra during val and test. Also, create these lists to collect all the losses

    collector_dict = {'gen_spec': {'validate': {}, 
                                   'test': {}},
                      'losses': {'train': {'total_loss': [], 'epsilon_loss': [], 'recons_loss': []}, 
                                 'val': [], 
                                 'test': []}} 

    for epoch in range(1, n_epoch+1):

        ### TRAINING LOOP ###

        # Set the epsilon in training mode
        cond_diffusion.epsilon.train()
        total_train_loss = 0 # Accumulate total train loss per epoch

        for _ in range(n_train_per_epoch):

            # Get the next batch
            try:
                batch = next(iterator_train)

            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The training dataset has been exhausted.")

            x0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index'] # Unpack the batch data
            x0= x0.to(device= cfg_train['device'])
            abundances= abundances.to(device= cfg_train['device']) # Move them to the device

            # Zero all the grads
            optimizer.zero_grad()

            # Add noise, denoise, and get x0_hat predictions
            x0_hat, xn, xn_hat = cond_diffusion.training_procedure(x0, abundances) 

            # Calculate the loss based on the reconstructed predictions and the actual spectra
            total_loss = loss(xn, xn_hat) 

            # Take the backprop and take a step
            total_loss.backward()
            optimizer.step()

            print(f'Epoch {epoch} | Batch {_ +1} | Total loss: {total_loss.item():.4f}')
            total_train_loss += total_loss.item()
            collector_dict['losses']['train']['total_loss'].append(total_loss.item())


        print(f"Epoch {epoch} | Total Training Loss: {total_train_loss:.4f}") # Training for this epoch finished, print the results

        ### VALIDATION STEP ###

        cond_diffusion.epsilon.eval() # Set the model in eval mode
        total_val_loss = 0 # Accumulate total validate loss

        # Create these lists to collect the generated stuff
        actual_list = [] 
        abundance_list = []
        generated_list = []

        with torch.no_grad():

            try:
                # Fetch the validation batch
                batch = next(iterator_validation)
            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The validation dataset has been exhausted.")
                
            x0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index'] # Unpack the batch data
            x0= x0.to(cfg_train['device'])
            abundances= abundances.to(cfg_train['device']) # Move them to the device

            # Get some x0_preds, conditional on the abundances themselves
            x0_pred, x_T = cond_diffusion.sample(x_T = None, ab= abundances)

            # Calculate the loss based on the reconstructed predictions and the actual spectra
            val_loss = loss.recons_loss(x0, x0_pred)
            collector_dict['losses']['val'].append(val_loss.item())
            total_val_loss += val_loss.item()

            # Append the actual, abundance, and generated spectra to the lists
            actual_list.append(x0.detach().cpu())
            abundance_list.append(abundances.detach().cpu())
            generated_list.append(x0_pred.detach().cpu())

            # After the validation epoch, store the generated spectra in the gen_spec dict, first turn them into tensors.
            collector_dict['gen_spec']['validate'][epoch] = {
                'actual': torch.cat(actual_list, dim=0),
                'abundances': torch.cat(abundance_list, dim=0),
                'generated': torch.cat(generated_list, dim=0)
            }

        print(f"Epoch {epoch} | Average Recons Loss in Validation: {total_val_loss:.4f}") # Validation for this epoch finished, print the results

    ### TESTING STEP ###

    cond_diffusion.epsilon.eval() # Set the model in eval (test) mode   
    total_test_loss = 0.0

    with torch.no_grad():

        try:
            # Fetch the test batch
            batch = next(iterator_test)
        except StopIteration:
            # If the dataset is exhausted, reset the iterator for the next epoch
            raise StopIteration("The test dataset has been exhausted.")
                
        x0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index'] # Unpack the batch data
        x0= x0.to(cfg_train['device'])
        abundances= abundances.to(cfg_train['device']) # Move them to the device

        # Get some x0_preds, conditional on the abundances themselves
        x0_pred, x_T = cond_diffusion.sample(x_T= None, ab= abundances)

        # Calculate the loss based on the reconstructed predictions and the actual spectra
        test_loss = loss.recons_loss(x0, x0_pred)
        total_test_loss += test_loss.item()

        # Append the actual, abundance, and generated spectra to the lists
        actual_list.append(x0.detach().cpu())
        abundance_list.append(abundances.detach().cpu())
        generated_list.append(x0_pred.detach().cpu())

        # After the test, store the generated spectra in the gen_spec dict, first turn them into tensors.
        collector_dict['gen_spec']['test'] = {
            'actual': torch.cat(actual_list, dim=0),
            'abundances': torch.cat(abundance_list, dim=0),
            'generated': torch.cat(generated_list, dim=0)
        }
        collector_dict['losses']['test'].append(test_loss.item())

        print(f"Average Test Loss: {total_test_loss:.4f}") # Test finished, print the results

    """
    Train, validation, and test has been finished. Now, final changes will be made to the necessary parts and outs will be given.
    """

    # Convert the lists to tensors
    collector_dict['losses']['train']['total_loss'] = torch.tensor(collector_dict['losses']['train']['total_loss'], dtype=torch.float32)
    collector_dict['losses']['train']['epsilon_loss'] = torch.tensor(collector_dict['losses']['train']['epsilon_loss'], dtype=torch.float32)
    collector_dict['losses']['train']['recons_loss'] = torch.tensor(collector_dict['losses']['train']['recons_loss'], dtype=torch.float32)
    collector_dict['losses']['val'] = torch.tensor(collector_dict['losses']['val'], dtype=torch.float32)
    collector_dict['losses']['test'] = torch.tensor(collector_dict['losses']['test'], dtype=torch.float32)

    return collector_dict, cond_diffusion # Return all the possibly useful stuff

def train_diffusion(cfg_train:(dict), cfg_export:(dict), diffusion_model:(torch.nn.Module), loss_fn:(torch.nn.Module), optimizer:(torch.optim), lr_scheduler, configured_data:(list[DataLoader])):

    """
    This function is to train the conditional diffusion model
    """

    # Unpack the relevant dicts
    device = cfg_train['device']
    dtype = cfg_train['dtype']
    n_epoch = cfg_train['n_epoch'] # Number of epochs
    n_tb_epoch = cfg_train['n_tb_epoch'] # Number of training instances/batches per epoch (this means if have B as batch, total train samples will be n_tb * B)

    # Get the iterators, the data unnormalizer etc.
    iterators, data_norm_dict = configured_data
    iter_train, iter_val, iter_test = iterators
    unnorm_lambda = get_unnormalizer(data_norm_dict)

    diffusion_model.to(device) # Move the epsilon and all the vars to the device

    # Setup the save paths
    model_save = cfg_export['model_save']
    if not model_save.endswith('.pth'): model_save += '.pth'
    val_save = cfg_export['test_save']
    if not val_save.endswith('.parquet'): val_save += '.parquet'
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
            x_0 = x_0.to(device); abundances = abundances.to(device)

            # Zero the grads
            optimizer.zero_grad()

            # Add noise, denoise, and get x_0 hat preds after internally augmenting the data (hat means pred)
            x_0_hat, x_0, x_n_hat, x_n = diffusion_model.training_procedure(x_0, abundances)
            # x_0 is augmented data, x_0_hat is the "fully recovered data", x_n is the added noise, x_n_hat is predicted noise

            # Calculate the loss
            train_loss, noise_loss, recons_loss = loss_fn(x_0_hat, x_0, x_n_hat, x_n)
            train_loss.backward() # Packprop the loss
            optimizer.step()

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
            x_0 = x_0.to(device); abundances = abundances.to(device)

            x_0_hat, x_T = diffusion_model.sample(abundances) # Pass the abundances to get a prediction for our spectrum

            val_loss = loss_fn.sample_loss(x_0_hat, x_0)
            wandb.log({
                "general/validation_average_loss": val_loss.item()
            })
            plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 100, epoch)

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
        x_0 = x_0.to(device); abundances = abundances.to(device)

        x_0_hat, x_T = diffusion_model.sample(abundances) # Pass the abundances to get a prediction for our spectrum

        test_loss = loss_fn.sample_loss(x_0_hat, x_0)

        wandb.log({
            "general/test_average_loss": test_loss.item()
        })
        plot_to_wandb(x_0, x_0_hat, abundances, name, orig_index, unnorm_lambda, 100, None) # Plot the testing results

    logging.info(f"Testing finished, Training of the Diffusion Model is Complete. The Difussion Model is saved at '{model_save}'")

