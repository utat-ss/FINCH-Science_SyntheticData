import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Iterator, Union


"""
Assumes as the input

cfg_data = {
    'cfg_import': {
        'data_handle': str for data handle,
        'spec_range': the spectral range, inclusive, as a list
    },
    'cfg_normalize': {
        'norm_type': type of normalization to be used (options: classic, dynamic, log, none)
    },
    'cfg_loader': {
        'n_epoch': n_epochs
        'n_val_epoch': n_vals per epoch
        'n_test': n_test data
        'n_train_batch': training data batch size
    }
}
"""

def get_vals(cfg_import:(dict), cfg_normalize:(dict)):

    """
    The function to get all the datasets.

    Args:
        cfg_import (dict): The dict config for the data imports
        cfg_normalize (dict): The dict to normalize the data

    Returns:
        spectra_tensor (torch.Tensor): Spectral val tensor
        abundances_tensor (torch.Tensor): Abundances tensor
        names (list): A list of all the spectral 
        indices (list): A list of the actual indices in a csv file
        norm_out (dict): Details related to how the dataset was normalized
    """

    # Unpack the dicts
    data_handle = cfg_import['data_handle']
    spec_range = cfg_import['spec_range']

    # Get the dataframe given the handle
    df = pd.read_csv(data_handle)

    # Get the spectral vals given the spectral range
    spectral_cols = [str(w) for w in range(spec_range[0], spec_range[1]+1, 10)]
    spectra = df[spectral_cols].values.astype("float32")
    spectra_tensor = torch.from_numpy(spectra)
    del spectra, spectral_cols

    norm_type = cfg_normalize['norm_type']
    if norm_type=='classic': # Classic, very blunt normalization
        spectra_tensor = (2 * spectra_tensor) - 1 # Bluntly normalize it
        norm_cfg = {
            'norm_type': norm_type
        }
    elif norm_type=='dynamic': # Dynamically scales using the max of the dataset
        max_vals = torch.max(spectra_tensor)
        min_vals = torch.min(spectra_tensor)
        spectra_tensor = 2 * ((spectra_tensor - min_vals) / (max_vals - min_vals)) - 1 # Sets the dataset [-1,1] filling all regions
        norm_out = {
            'norm_type': norm_type,
            'max_vals': max_vals,
            'min_vals': min_vals
        }
    elif norm_type =='log': # Dynamically scales using log and min and max of the dataset
        eps = 1e-6
        spectra_tensor = torch.log(spectra_tensor + eps)
        max_vals = torch.max(spectra_tensor)
        min_vals = torch.min(spectra_tensor)
        spectra_tensor = 2 * ((spectra_tensor - min_vals) / (max_vals - min_vals)) - 1
        norm_out = {
            'norm_type': norm_type,
            'max_vals': max_vals,
            'min_vals': min_vals,
            'eps': eps
        }
    elif norm_type=='none':
        norm_out = {
            'norm_type': norm_type
        }

    # Get the abundances
    abundances = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
    abundances_tensor = torch.from_numpy(abundances)
    del abundances

    # Get the spectral names
    names = df['Spectra'].to_list()

    # Get the original indices
    indices = range(len(names))

    return spectra_tensor, abundances_tensor, names, indices, norm_out

class HyperSpectralDataset(Dataset):
    """
    Defining this class so that we can keep track of spectra, abundances, names, and indices.

    Args:
        spectra (torch.Tensor): Tensor of all the spectra, as a tensor
        abundances (torch.Tensor): Abundances of all the spectra, as a tensor
        names (list): Names of the spectra, as a list
        indices (list): Indices in order, as a list
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
            'names': self.names[idx],
            'orig_index': self.indices[idx]
        }

def get_dataloaders(ds: HyperSpectralDataset, cfg_loader: dict) -> list[DataLoader]:
    """
    Gets the dataloaders for train, test, validate datasets.

    Args:
        ds (HyperSpectralDataset): The entire dataset as a HyperSpectralDataset class.
        cfg_loader (dict):
            n_epoch (int): How many epochs
            n_val_epoch (int): How many validation points per epoch
            n_test (int): How many test samples in total
            n_train_batch (int): Train dataloader batch size
    
    Returns:
        dataloaders (list[DataLoader]): A list of the prepared dataloaders
    """

    # Take in the values from the dict
    n_epoch = cfg_loader.get('n_epoch', 50)
    n_val_epoch = cfg_loader.get('n_val_epoch', 4)
    n_test = cfg_loader.get('n_test', 123)
    n_train_batch = cfg_loader.get('n_train_batch', 5)

    # Infer the amount of n_train, ensure completeness
    n_train = int(len(ds) - n_epoch*n_val_epoch - n_test)

    # Separate the dataset into validate and temporary
    ds_test, ds_temp = random_split(ds, [n_test, n_train + n_val_epoch*n_epoch]) 

    # Separate the temp dataset into train and test
    ds_train, ds_validate = random_split(ds_temp, [n_train, n_val_epoch*n_epoch])

    dataloaders = [DataLoader(ds_train, batch_size=n_train_batch, shuffle=True), DataLoader(ds_validate, batch_size=n_val_epoch, shuffle= True), DataLoader(ds_test, batch_size=n_test, shuffle=True)]

    return  dataloaders # Make dataloaders into a list and ship them

def get_inf_iterators(dataloaders:(Union[list[DataLoader], DataLoader])) -> list[Iterator]:
    """
    Gets the infinite iterators, they can be infinitely iterated through. Usually only pass 

    Args:
        dataloaders (list[DataLoader] or DataLoader): A list of the already prepared dataloaders or only a single one
    
    Returns:
        inf_iterators (list[Iterator]): A list of infinitely callable iterators, they cycle through the data
    """
    def cycle(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    if isinstance(dataloaders, DataLoader): # Check if the input is only a single dataloader if so, define it as a list
        dataloaders = [dataloaders]
    return [cycle(dl) for dl in dataloaders]

def get_data(cfg_data:(dict)):

    """
    Gets all the data on its own.
    """

    cfg_import = cfg_data['cfg_import'] # Get the dicts to input to get_vals
    cfg_normalize = cfg_data['cfg_normalize']

    spectra_tensor, abundances_tensor, names, indices, norm_out = get_vals(cfg_import, cfg_normalize) # Get the vals

    ds = HyperSpectralDataset(spectra_tensor, abundances_tensor, names, indices) # Create a ds using the vals

    cfg_loader = cfg_data['cfg_loader'] # Get the dict to input to get_dataloaders
    dataloaders = get_dataloaders(ds, cfg_loader) # Get the dataloaders as a list

    iterators = [
        *get_inf_iterators(dataloaders[0]),
        iter(dataloaders[1]),
        iter(dataloaders[2])
    ]

    del dataloaders, spectra_tensor, abundances_tensor, names, indices

    return [iterators, norm_out]






