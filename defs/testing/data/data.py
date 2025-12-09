from .structure import *

from typing import Union

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

def vals_from_csv(save_path:(str), spec_range:(list[int])):

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
    This function takes in the data config, and then creates the dataloaders/returns them
    """

    cfg_normalize = cfg_data['cfg_normalize']
    spec_range = cfg_data['cfg_import']['spec_range']

    
    path_ksi_train = cfg_data['cfg_import']['path_ksi_train']
    path_psi2 = cfg_data['cfg_import']['path_psi2']







    pass