"""
The code here serves the purpose of taking the data separating it into to be synthesized
and real data
"""

import torch
import pandas as pd

from math import floor

def vals_from_csv(save_path:(str), spec_range:(list[int])) -> torch.Tensor | torch.Tensor | list[str] | list[int]:
    """
    Takes in the save path and necessary spectral range, returns tensors of spectra and abundances

    Args:
        save_path (str): The path csv was saved to, in this case, psi2
        spec_range (list[int]): An inclusive double entry list of spec range ints

    Returns:
        spectra_tensor (torch.Tensor): A tensor of spectra, with shape [n_spectra, n_bands]
        abundances_tensor (torch.Tensor): A tensor of abundances, with shape [n_spectra, n_abundances]
        names (list[str]): List of all string spectral names, len=n_spectra
        indices (list[int]): List of all true spectral integer indices, len=n_spectra
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

def separate_data(spectra_tensor:(torch.Tensor), abundances_tensor:(torch.Tensor), names:(list[str]), indices:(list[int]), separation_ratio:(float)=0.5, verbose:(bool)=False):
    """
    Separates the spectra_tensor, abundances_tensor (if verbose=True), names (if verbose=True), indices into two parts each, given the separation_ratio

    Args:
        spectra_tensor (torch.Tensor): A tensor of spectra, with shape [n_spectra, n_bands]
        abundances_tensor (torch.Tensor): A tensor of abundances, with shape [n_spectra, n_abundances]
        names (list[str]): List of all string spectral names, len=n_spectra
        indices (list[int]): List of all true spectral integer indices, len=n_spectra
        
    Returns:
        separated tensors and lists in the order as args, in a list, starting from real -> to be synthesized
    """

    separation_len = floor(separation_ratio * len(names))

    spectra_real = spectra_tensor[:separation_len, :]
    abundances_real = abundances_tensor[:separation_len, :]
    spectra_synth = spectra_tensor[separation_len:, :]
    abundances_synth = abundances_tensor[separation_len:, :]

    if verbose:
        names_real = names[:separation_len]
        indices_real = indices[:separation_len]
        names_synth = names[separation_len:]
        indices_synth = indices[separation_len:]
        return spectra_real, abundances_real, names_real, indices_real, spectra_synth, abundances_synth, names_synth, indices_synth, separation_len
    else:
        return spectra_real, abundances_real, None, None, spectra_synth, abundances_synth, None, None, separation_len


