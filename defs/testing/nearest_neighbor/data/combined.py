"""
The code included here is to combine the synthetic and real data, and make it into a usable form 
for the nearest neighborhood calculations.
"""
import torch

from .get_data import vals_from_csv, separate_data
from .synthetic import synthesize_from_abundances

def get_combined_data(cfg_import:(dict), cfg_synthesis:(dict)) -> torch.Tensor | torch.Tensor | list[str] | list[int] | int:
    """
    Conducts the entire operation of reading data, synthesizing parts of it, and combining it

    Args:
        cfg_import (dict): The dict of import configurations
        cfg_synthesis (dict): Specifics of the synthesis process
        
    Returns:
        combined_spectra (torch.Tensor): The combined spectra where [:separation_len] are real data, rest is synthesized
        combined_abundances (torch.Tensor): Same as combined_spectra, but for abundances
        combined_names (list[str]): The list of combined names
        combined_indices (list[int]): The list of combined true indices
        separation_len (int): How much of the data has been taken in as real data, to not be touched, rest are synthesized
    """
    unnormalizer = cfg_synthesis['unnormalizer']

    # Gets the actual tensors
    spectra_tensor, abundances_tensor, names, indices = vals_from_csv(cfg_import['psi2_path'], cfg_import['spec_range'])

    # Separates the actual tensors into real and to be synthesized part
    spectra_real, abundances_real, names_real, indices_real, spectra_tobe_synth, abundances_synth, names_synth, indices_synth, separation_len = separate_data(spectra_tensor, abundances_tensor, names, indices, cfg_import['separation_ratio'], verbose=True)

    # Synthesizes some data using the abundances allocated for synthesis
    spectra_synth, _ = synthesize_from_abundances(cfg_synthesis, abundances_synth)

    spectra_real = unnormalizer(spectra_real)
    spectra_synth = unnormalizer(spectra_synth)

    # Combines all of the results
    data_real = [spectra_real, abundances_real, names_real, indices_real]
    data_synth = [spectra_synth, abundances_synth, names_synth, indices_synth]

    return data_real, data_synth, separation_len