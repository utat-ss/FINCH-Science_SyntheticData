import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Iterator, Union
import math


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

    norm_type = cfg_normalize['norm_type']
    if norm_type=='classic': # Classic, very blunt normalization
        spectra_tensor = (2 * spectra_tensor) - 1 # Bluntly normalize it
        norm_out = {
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
    elif norm_type == 'statistical': # Standardizes to Zero Mean, Unit Variance
        mean_vals = torch.mean(spectra_tensor)
        std_vals = torch.std(spectra_tensor)
        
        # Standardize: (X - mu) / sigma
        # 1e-8 is added to prevent division by zero if a feature is constant
        spectra_tensor = (spectra_tensor - mean_vals) / (std_vals + 1e-8)
        
        norm_out = {
            'norm_type': norm_type,
            'mean_vals': mean_vals,
            'std_vals': std_vals
        }
    elif norm_type=='none':
        norm_out = {
            'norm_type': norm_type
        }
    else: raise ValueError(f"Unknown/Unsupported norm_type: {norm_type}")

    # Get the abundances
    abundances = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
    abundances_tensor = torch.from_numpy(abundances)
    del abundances

    # Get the spectral names
    names = df['Spectra'].astype(str).to_list()

    # Get the original indices
    indices = list(range(len(names)))

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
    def __init__(self, spectra:(torch.Tensor), abundances:(torch.Tensor), names:(list[str]), indices:(list[int])):

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
    
    def save_to_csv(self, save_path:(str), spec_range:(list[int])):

        """
        Saves the normalized version of the HyperSpectralDataset into a given save path as a csv.

        Args:
            self (included): Contains the spectrum, abundances, names, orig_index
            save_path (str): The save path of the dataset (must not have .csv at the end)
            norm_type (str): The normalization type of the given dataset
            spec_range (list[int]): An int list of two entries with inclusive spectral range

        Returns:
            saved (.csv): A csv file with original indices, names, abundances, and the normalized spectrum at path 'path'_'norm_type'.csv
        """

        import pandas as pd

        if not save_path.endswith('.csv'):  save_path += '.csv' # Control the save path

        df = pd.DataFrame()

        df['orig_index'] = self.indices
        df['Spectra'] = self.names
        df[['gv_fraction', 'npv_fraction', 'soil_fraction']] = self.abundances.detach().cpu().numpy()
        spectral_cols = [w for w in range(spec_range[0], spec_range[1]+1, 10)]
        spectra_df = pd.DataFrame(
            self.spectra.detach().cpu().numpy(),
            columns=spectral_cols,
            index=df.index  # Critical: ensures rows match up
        )
        df = pd.concat([df, spectra_df], axis=1)

        df.sort_values(by='orig_index', inplace=True)
        df.to_csv(save_path, index=False)

def save_split_wrapper(subset, save_path, spec_range):
    """
    Takes a subset of HyperSpectralDataset, converts it to a full HyperSpectralDataset, and then saves it
    """

    parent_ds = subset.dataset
    split_indices = subset.indices

    sub_spectra = parent_ds.spectra[split_indices]
    sub_abundances = parent_ds.abundances[split_indices]
    sub_names = [parent_ds.names[i] for i in split_indices]
    sub_orig_indices = [parent_ds.indices[i] for i in split_indices]

    temp_ds = HyperSpectralDataset(
        spectra=sub_spectra, 
        abundances=sub_abundances, 
        names=sub_names, 
        indices=sub_orig_indices
    )

    temp_ds.save_to_csv(save_path, spec_range)

def get_dataloaders(ds:(HyperSpectralDataset), cfg_loader:(dict), cfg_dataset_save:(dict)) -> list[DataLoader]:
    """
    Gets the dataloaders for train, test, validate datasets.

    Args:
        ds (HyperSpectralDataset): The entire dataset as a HyperSpectralDataset class.
        cfg_loader (dict):
            n_val (int): How many validation points in total
            n_test (int): How many test samples in total
            n_train_batch (int): Train dataloader batch size
            num_workers (int): Num workers for the training dataloader
            prefetch_factor (int): Prefetch factor for the training dataloader; pre-loaded batches = num_workers * prefetch_factor
            seed (int): Seed for random split
        cfg_dataset_save (dict):
            psi1_path (str): Save path for the psi1 dataset (train of simpler_data)
            psi2_path (str): Save path for the psi2 dataset (val+test of simpler data)
            spec_range (list[int]): the spectral range, inclusive, as a list
            norm_type (str): Type of normalization applied to spectra

    Returns:
        dataloaders (list[DataLoader]): A list of the prepared dataloaders
    """

    # Take in the important values from the loader
    n_val = cfg_loader.get('n_val', 200)
    n_test = cfg_loader.get('n_test', 123)
    n_train_batch = cfg_loader.get('n_train_batch', 5)
    num_workers_train = cfg_loader.get('num_workers', 4)
    prefetch_factor_train = cfg_loader.get('prefetch_factor', 2)

    # Infer the amount of n_train, ensure completeness
    n_train = int(len(ds) - n_val - n_test)

    # Get the RNG, for reproducibility
    generator = torch.Generator()
    generator.manual_seed(cfg_loader['seed'])

    # Separate the dataset into train and temporary
    ds_train, ds_temp = random_split(ds, [n_train, n_test + n_val], generator) 
    save_split_wrapper(ds_train, cfg_dataset_save['psi1_path'], cfg_dataset_save['spec_range'])
    save_split_wrapper(ds_temp, cfg_dataset_save['psi2_path'], cfg_dataset_save['spec_range'])

    # Separate the temp dataset into train and test
    ds_test, ds_validate = random_split(ds_temp, [n_test, n_val], generator)

    dataloaders = [DataLoader(ds_train, batch_size=n_train_batch, generator=generator, shuffle=True, drop_last=False, num_workers=num_workers_train, persistent_workers=True, prefetch_factor=prefetch_factor_train, pin_memory=True), DataLoader(ds_validate, batch_size=n_val, shuffle=False), DataLoader(ds_test, batch_size=n_test, shuffle=False)]

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
    cfg_dataset_save = cfg_data['cfg_dataset_save']; cfg_dataset_save['spec_range'] = cfg_import['spec_range']; cfg_dataset_save['norm_type'] = norm_out['norm_type']
    dataloaders = get_dataloaders(ds, cfg_loader, cfg_dataset_save) # Get the dataloaders as a list

    iterators = [
        *get_inf_iterators(dataloaders[0]),
        *get_inf_iterators(dataloaders[1]),
        iter(dataloaders[2])
    ]

    del dataloaders, spectra_tensor, abundances_tensor, names, indices

    return [iterators, norm_out]

def get_unnormalizer(data_norm_dict:(dict)):
    """
    The function to get the unnormalizer function of a given data normalizer.
    """

    if data_norm_dict['norm_type'] == 'classic':

        return lambda normed_data: (normed_data+1)/2
    
    elif data_norm_dict['norm_type'] == 'dynamic':

        max_vals, min_vals = data_norm_dict['max_vals'], data_norm_dict['min_vals']

        # Ensure they are tensors (if they are floats, this does nothing which is fine)
        if not torch.is_tensor(max_vals): max_vals = torch.tensor(max_vals)
        if not torch.is_tensor(min_vals): min_vals = torch.tensor(min_vals)

        return lambda normed_data: ((normed_data + 1) * (max_vals.to(normed_data.device) - min_vals.to(normed_data.device))) / 2 + min_vals.to(normed_data.device)
    
    elif data_norm_dict['norm_type'] == 'log':

        max_vals, min_vals, eps = data_norm_dict['max_vals'], data_norm_dict['min_vals'], data_norm_dict['eps']

        if not torch.is_tensor(max_vals): max_vals = torch.tensor(max_vals)
        if not torch.is_tensor(min_vals): min_vals = torch.tensor(min_vals)
        if not torch.is_tensor(eps): eps = torch.tensor(eps)

        return lambda normed_data: torch.exp(((normed_data + 1)*(max_vals.to(normed_data.device) - min_vals.to(normed_data.device)))/2 + min_vals) + eps.to(normed_data.device)
    
    if data_norm_dict['norm_type'] == 'statistical':

        mean_vals = data_norm_dict['mean_vals']
        std_vals = data_norm_dict['std_vals']
        
        if not torch.is_tensor(mean_vals): mean_vals = torch.tensor(mean_vals)
        if not torch.is_tensor(std_vals): std_vals = torch.tensor(std_vals)

        return lambda normed_data: (normed_data * std_vals.to(normed_data.device)) + mean_vals.to(normed_data.device)
    
    elif data_norm_dict['norm_type']=='none':

        return lambda normed_data: normed_data
    
    else: raise ValueError(f"Unknown/Unsupported normalization type {data_norm_dict['norm_type']}.")



