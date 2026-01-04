import torch
import pandas as pd

from torch.utils.data import DataLoader, random_split
from .structure import *

from typing import Union, Iterator

"""

This dataset pipeline assumes:



All the docstrings follow the assumption:

psi (full, lab dataset) = Union( psi_1 (train of synthesizer) + psi_2 (val/test of synthesizer) )

What we want to do is:

1 - set the training data for critic, ksi_train = Synthesizer(psi_1). Essentially using the majority of synthesized dataset.
    if using baseline, ksi_train = psi_1

2 - set the ksi_val = Subsection(psi_2), i.e. it comes from a subsection of the real data that was not used for synthesizer train

3 - set the ksi_test = Subsection(psi_2), i.e. it comes from a subsection of the real data that was not used for synthesizer train

cfg_data:

    cfg_normalize:
        norm_type
    
    cfg_import:
        spec_range
        path_ksi_train 
        path_psi2
    
    seed
"""

def vals_from_csv(save_path:(str), spec_range:(list[int])) -> torch.Tensor | torch.Tensor | list[str] | list[int]:

    """
    Gets the vals from a csv file that was exported via dataset creation in the training of synthesizers.

    Args:
        save_path (str): The path csv was saved to
        spec_range (list[int]): An inclusive double entry list of spec range ints
    
    Returns:
        spectra_tensor (torch.Tensor): Tensor of all the specra in csv
        abundances_tensor (torch.Tensor): Tensor of all the abundances in csv
        names (list[str]): List of all string spectral names
        indices (list[int]): List of all true spectral integer indices
    """
    df = pd.read_csv(save_path)

    spectral_cols = [str(w) for w in range(spec_range[0], spec_range[1]+1, 10)]
    spectra = df[spectral_cols].values.astype("float32")
    spectra_tensor = torch.from_numpy(spectra)

    abundances = df[["gv_fraction","npv_fraction","soil_fraction"]].values.astype("float32")
    abundances_tensor = torch.from_numpy(abundances)

    # Get the spectral names
    names = df['Spectra'].to_list()

    # Get the original indices
    indices = df['orig_index'].to_list()

    return spectra_tensor, abundances_tensor, names, indices

def get_data(cfg_import:(dict), cfg_loader:(dict)) -> list[Iterator]:

    """
    This function takes in the data config, and then creates the dataloaders/returns them

    Args:
        cfg_import (dict): Import specifics of ksi_train and ksi_val
        cfg_loader (dict): Loader specifics

    Returns:
        list(Iterator): A list where [0] is finite loader for train (ksi_train), [1] is infinite loader for val (ksi_val), [2] is finite loader for test (ksi_test)
    """
    # Unpack necessary stuff
    path_ksi_train = cfg_import['path_ksi_train']
    path_psi2 = cfg_import['path_psi2']
    spec_range = cfg_import['spec_range']

    # Initialize the generator
    generator = torch.Generator()
    generator.manual_seed(cfg_loader['seed'])

    # Create the datasets ds_ksi_train and ds_psi2
    spectra, abundances, names, indices = vals_from_csv(path_ksi_train, spec_range)
    ds_ksi_train = HyperSpectralDataset(spectra, abundances, names, indices)
    spectra, abundances, names, indices = vals_from_csv(path_psi2)
    ds_psi2 = HyperSpectralDataset(spectra, abundances, names, indices)
    del spectra, abundances, names, indices

    # Seperate ds_psi2 into ds_ksi_val, ds_ksi_test
    ds_ksi_val, ds_ksi_test = random_split(ds_psi2, [cfg_loader['n_val'], cfg_loader['n_test']], generator)

    batch_size = cfg_loader['n_train_batch']; num_workers = cfg_loader['num_workers']; prefetch_factor = cfg_loader['prefetch_factor']
    dl_ksi_train = DataLoader(ds_ksi_train, batch_size=batch_size, generator=generator, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor, pin_memory=True)
    dl_ksi_val = DataLoader(ds_ksi_val, batch_size=cfg_loader['n_val'], shuffle=False)
    dl_ksi_test = DataLoader(ds_ksi_test, batch_size=cfg_loader['n_test'], shuffle=False)

    return [iter(dl_ksi_train), get_inf_iterator(dl_ksi_val), iter(dl_ksi_test)]