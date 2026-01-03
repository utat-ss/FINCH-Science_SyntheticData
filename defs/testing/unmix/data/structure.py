r"""
All the docstrings follow the assumption:

psi (full, lab dataset) = Union( psi_1 (train of synthesizer) + psi_2 (val/test of synthesizer) )

What we want to do is:

1 - set the training data for critic, ksi_train = Synthesizer(psi_1). Essentially using the majority of synthesized dataset.
2 - set the ksi_val = Subsection(psi_2), i.e. it comes from a subsection of the real data that was not used for synthesizer train
3 - set the ksi_test = Subsection(psi_2), i.e. it comes from a subsection of the real data that was not used for synthesizer train
"""



import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from typing import Iterator

def get_vals(data_handle:(str), spec_range:(list[int]), cfg_normalize:(dict)):

    """
    The function to get all the datasets.

    Args:
        data_handle (str): The str path to the data
        spec_range (list[int]): Inclusive range of the spectral range given as list of ints
        cfg_normalize (dict): The dict to normalize the data

    Returns:
        spectra_tensor (torch.Tensor): Spectral val tensor
        abundances_tensor (torch.Tensor): Abundances tensor
        names (list): A list of all the spectral 
        indices (list): A list of the actual indices in a csv file
        norm_out (dict): Details related to how the dataset was normalized
    """

    # Unpack the dicts

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
    def __init__(self, spectra:(torch.Tensor), abundances:(torch.Tensor), names:(list), indices:(list)):

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
    

def get_dataloader(ds:(HyperSpectralDataset), cfg_loader:(dict), seed:(int), ds_type:(str), ds_use:(str)) -> DataLoader:
    """
    Gets the dataloaders for train, test, validate datasets.

    Args:
        ds (HyperSpectralDataset): The entire dataset as a HyperSpectralDataset class.
        cfg_loader (dict):
            n_epoch (int): How many epochs in total
            n_batch_epoch (int): How many batches in total, in a single epoch
            batch_size (int): Size of each batch
        seed (int): Seed for random shuffle/split
        ds_type (dict): Type of the data, either 'synthesized' or 'real'
        ds_use (dict): The use of the dataset, either 'train', 'val', 'test'
    
    Returns:
        dataloaders (DataLoader): The dataloader
    """

    # Get the RNG, for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    if (ds_type == 'synthesized' or ds_type == 'real') and ds_use == 'train':

        # Unpack the dict
        n_epoch = cfg_loader['n_epoch']
        n_batch_epoch = cfg_loader['n_batch_epoch']
        batch_size = cfg_loader['batch_size']

        assert len(ds) >= n_epoch*n_batch_epoch*batch_size

        return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)
    
    
    if ds_type == 'real' and (ds_use == 'val' or ds_use == 'test'):

        # No need to unpack anything

        return DataLoader(ds, batch_size=len(ds), shuffle=True, generator=generator)
    
    else:
        raise ValueError(f"Unknown combination of dataset use: {ds_use} and dataset type: {ds_type}")


def get_inf_iterator(dataloader:(DataLoader)) -> Iterator:
    """
    Gets the infinite iterator, it can be infinitely iterated through.

    Args:
        dataloaders (DataLoader): An already prepared dataloader
    
    Returns:
        cycle(dataloader): An infinitely iterable dataloader
    """
    def cycle(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    return cycle(dataloader)