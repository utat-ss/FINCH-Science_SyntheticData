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

class HyperSpectralDataset(Dataset):
    """
    Defining this class so that we can keep track of spectra, abundances, names, and indices.

    Args:
        spectra (torch.Tensor): Tensor of all the spectra, as a tensor
        abundances (torch.Tensor): Abundances of all the spectra, as a tensor
        names (list[str]): Names of the spectra, as a list
        indices (list[int]): Indices in order, as a list
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

def get_inf_iterator(dataloader:(DataLoader)) -> Iterator:
    """
    Gets the infinite iterator, it can be infinitely iterated through.

    Args:
        dataloader (DataLoader): An already prepared dataloader
    
    Returns:
        cycle(dataloader): An infinitely iterable dataloader
    """
    def cycle(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    return cycle(dataloader)