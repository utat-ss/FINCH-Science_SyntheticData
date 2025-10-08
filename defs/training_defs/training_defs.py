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
            - 'validate' (int): How many samples to be used to validate at the end
            - 't_batch' (int): Train with batches of what size, every epoch
            - 'test' (int): How many samples to test on each epoch
            - 'epoch' (int): How many epochs in total
    """

    # Take in the values from the dict
    n_validate = cfg_loader.get('validate', 23)
    n_test = cfg_loader.get('test', 4)
    n_t_batch = cfg_loader.get('t_batch', 1)
    n_epoch = cfg_loader.get('epoch', 50)

    # Infer the amount of n_train, ensure completeness
    n_train = int(len(ds) - n_epoch*n_test - n_validate)
    assert n_train % n_epoch == 0, f"Something is wrong with the sample allocation, n_train: {n_train} is not divisible by n_epoch: {n_epoch}"
    assert n_train/n_epoch % n_t_batch, f"Train sample per epoch: {n_train/n_epoch}, must be divisible by batch size in train: {n_t_batch}"

    # Separate the dataset into validate and temporary
    ds_validate, ds_temp = random_split(ds, [n_validate, n_train + n_test*n_epoch]) 

    # Separate the temp dataset into train and test
    ds_train, ds_test = random_split(ds_temp, [n_train, n_epoch*n_test])

    return DataLoader(ds_train, batch_size=n_t_batch, shuffle=True), DataLoader(ds_test, batch_size=1, shuffle= True), DataLoader(ds_validate, batch_size=1, shuffle=True)

def train_diffusion(cfg_train: dict, loss, cond_diffusion, data_handle: str=None):

    """
    The function to train and validate the models, takes in:
        - cfg_train (dict): 
            - cfg_loader (dict):
                - 'validate' (int): How many samples to be used to validate at the end
                - 't_batch' (int): How many train batches in total, to be trained on
                - 'test' (int): How many samples to test on each epoch
                - epoch' (int): How many epochs in total
            - 'range' (list): Spectral range [lower, upper]
        - loss (fn): Loss function, must have MSE and SAM at the very least
        - cond_diffusion (class): The conditional diffusion class, already initialized
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
        'cfg_loader', {'validate': 23, 't_batch': 1, 'test': 4, 'epoch': 50}
    ) # Get the configs for the loader first

    # Get the dataloaded datasets
    ds_train, ds_test, ds_validation = get_dataloaders(ds, cfg_loader=cfg_loader)

    n_validate = cfg_loader.get('validate', 23)
    n_epoch = cfg_loader.get('epoch', 50)
    n_t_batch = cfg_loader.get('t_batch', 1)
    n_test = cfg_loader.get('test', 4)

    for epoch in range(1, n_epoch+1):

        cond_diffusion.epsilon.train()

        # The rest will be pushed soon

    
