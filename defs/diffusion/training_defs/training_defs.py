"""

This is the file to define all the training functions. It includes:
    - Diffusion training function
    - 

"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

def get_vals(data_handle: str= r'data\simpler_data_rwc.csv', spec_range: list[int]= [400, 2490]):

    """
    The function to get all the datasets.

    Parameters:
        - data_handle (str): The dataset handle, path str rooting the github repo
        - range (list[int]): A list of spectral limits

    Outputs:
        - spectra_tensor (torch.Tensor): Spectral val tensor
        - abundances_tensor (torch.Tensor): Abundances tensor
        - names (list): A list of all the spectral 
        - indices (list): A list of the actual indices in a csv file
    """

    # Get the dataframe given the handle
    df = pd.read_csv(data_handle)

    # Get the spectral vals given the spectral range
    spectral_cols = [str(w) for w in range(spec_range[0], spec_range[1]+1, 10)]
    spectra = df[spectral_cols].values.astype("float32")
    spectra_tensor = torch.from_numpy(spectra)
    del spectra, spectral_cols

    # Get the abundances
    abundances = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
    abundances_tensor = torch.from_numpy(abundances)
    del abundances

    # Get the spectral names
    names = df['Spectra'].to_list

    # Get the original indices
    indices = range(len(names))

    return spectra_tensor, abundances_tensor, names, indices

class HyperSpectralDataset(Dataset):

    """
    Defining this class so that we can keep track of spectra, abundances, names, and indices.

    Parameters:
        - spectra (torch.Tensor): Tensor of all the spectra, as a tensor
        - abundances (torch.Tensor): Abundances of all the spectra, as a tensor
        - names (list): Names of the spectra, as a list
        - indices (list): Indices in order, as a list

    """
    def __init__(self, spectra: torch.Tensor, abundances: torch.Tensor, names: list, indices: list):

        self.spectra = spectra
        self.abundances = abundances
        self.names = names
        self.indices = indices

        def __len__(self):
            return len(self.names)
        
        def __getitem__(self, idx):

            return {
                'spectrum': self.spectra[idx],
                'abundances': self.abundances[idx],
                'name': self.names[idx],
                'orig_index': self.indices[idx]
            }

def get_dataloaders(ds: HyperSpectralDataset, cfg_loader: dict):

    """
    Gets the dataloaders for train, test, validate datasets.

    Parameters:
        - ds (HyperSpectralDataset): The entire dataset as a HyperSpectralDataset class.
        - cfg_loader (dict):
            - 'test' (int): How many samples to be used to test at the end
            - 't_batch' (int): Train with batches of what size, every epoch
            - 'validate' (int): How many samples to validate on each epoch
            - 'epoch' (int): How many epochs in total
    """

    # Take in the values from the dict
    n_test = cfg_loader.get('test', 23)
    n_validate = cfg_loader.get('validate', 4)
    n_t_batch = cfg_loader.get('t_batch', 1)
    n_epoch = cfg_loader.get('epoch', 50)

    # Infer the amount of n_train, ensure completeness
    n_train = int(len(ds) - n_epoch*n_validate - n_test)
    assert n_train % n_epoch == 0, f"Something is wrong with the sample allocation, n_train: {n_train} is not divisible by n_epoch: {n_epoch}"
    assert n_train/n_epoch % n_t_batch, f"Train sample per epoch: {n_train/n_epoch}, must be divisible by batch size in train: {n_t_batch}"

    # Separate the dataset into validate and temporary
    ds_test, ds_temp = random_split(ds, [n_test, n_train + n_validate*n_epoch]) 

    # Separate the temp dataset into train and test
    ds_train, ds_validate = random_split(ds_temp, [n_train, n_epoch*n_validate])

    return DataLoader(ds_train, batch_size=n_t_batch, shuffle=True), DataLoader(ds_validate, batch_size=1, shuffle= True), DataLoader(ds_test, batch_size=1, shuffle=True)

def train_diffusion(cfg_train: dict, cond_diffusion, loss, optimizer: torch.optim, data_handle: str=None):

    """
    The function to train and validate the models, takes in:
        - cfg_train (dict): 
            - cfg_loader (dict):
                - 'test' (int): How many samples to be used to test at the end
                - 't_batch' (int): How many train batches in total, to be trained on
                - 'validate' (int): How many samples to validate on each epoch
                - 'epoch' (int): How many epochs in total
            - 'range' (list): Spectral range [lower, upper]
            - 'device' : The device being used
        - cond_diffusion (class): The conditional diffusion class, already initialized
        - loss (class): Loss function class, must have MSE and SAM at the very least
        - optimizer (torch.optim): The optimizer to use
        - data_handle (str): The path handle to the dataset
    """

    # First, we will pre-process and parse all the data
    # Get the spectral range that we want to use
    spec_range = cfg_train.get('range', [400, 2490])

    # Parse the csv into sub-parts we want to use
    spectra, abundances, names, indices = get_vals(data_handle=data_handle, spec_range=spec_range)

    # Generate the hyperspectral dataset
    ds = HyperSpectralDataset(spectra=spectra, abundances=abundances, names=names, indices=indices)

    # Dataload the giant dataset into train, test, validation datasets
    cfg_loader = cfg_train.get(
        'cfg_loader', {'test': 23, 't_batch': 1, 'validate': 4, 'epoch': 50}
    ) # Get the configs for the loader first

    # Get the dataloaded datasets
    ds_train, ds_validation , ds_test= get_dataloaders(ds, cfg_loader=cfg_loader)

    n_validate = cfg_loader.get('test', 23)
    n_epoch = cfg_loader.get('epoch', 50)
    n_t_batch = cfg_loader.get('t_batch', 1)
    n_test = cfg_loader.get('validate', 4)

    # Infer the amount of n_train, ensure completeness
    n_train = int(len(ds) - n_epoch*n_validate - n_test)
    n_train_per_epoch = n_train // n_epoch

    for epoch in range(1, n_epoch+1):

        # Set the epsilon in training mode
        cond_diffusion.epsilon.train()
        total_train_loss = 0 # Accumulate total train loss per epoch
        
        for _ in range(n_train_per_epoch):

            # Get the next batch
            try:
                batch = next(ds_train)

            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The training dataset has been exhausted.")

            x0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index'] # Unpack the batch data
            x0.to(cfg_train['device']), abundances.to(cfg_train['device']) # Move them to the device

            # Zero all the grads
            optimizer.zero_grad()

            # Add noise, denoise, and get x0_hat predictions
            x0_hat, xn, xn_hat = cond_diffusion.training_procedure(x0, abundances) 

            # Calculate the loss based on the reconstructed predictions and the actual spectra
            total_loss = loss.calc(x0, x0_hat, xn, xn_hat) 

            # Take the backprop and take a step
            total_loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch + 1} | Average Training Loss: {total_train_loss:.4f}") # Training for this epoch finished, print the results

        cond_diffusion.epsilon.eval() # Set the model in eval mode
        total_val_loss = 0 # Accumulate total validate loss

        with torch.no_grad():

            for _ in range(n_validate):

                try:
                    # Fetch the validation batch
                    batch = next(val_loader)
                except StopIteration:
                    val_loader = iter(DataLoader(ds_validation, batch_size=1, shuffle=False))
                    batch = next(val_loader)
                
                x0, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index'] # Unpack the batch data
                x0.to(cfg_train['device']), abundances.to(cfg_train['device']) # Move them to the device

                # Get some x0_preds, conditional on the abundances themselves
                x0_pred = cond_diffusion.sample(ab= abundances)

                









        # The rest will be pushed soon


    
