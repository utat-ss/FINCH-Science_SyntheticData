"""
This is the code to take in abundances from psi2 and then synthesize some data given those abundances
"""

from ....synthesis.loaders import load_model
from ....synthesis.lean.lean_synth import *

def synthesize_from_abundances(cfg_synthesis:(dict), ab_tensor:(torch.Tensor)) -> torch.Tensor:
    """
    This function synthesizes spectra given the abundance tensor, using the cfg_synthesis and cfg_model to get the synthesizer

    Args:
        cfg_synthesis (dict): The dict that contains synthesis specifics
        ab_tensor (torch.Tensor): The abundance tensor to synthesize from, shape [n_spectra, n_abundances]

    Returns:
        synthesized_spectra (torch.Tensor): The synthesized spectra tensor, shape [n_synth_spectra, n_bands]
        ab_tensor (torch.Tensor): The input abundance tensor, shape [n_synth_spectra, n_abundances]
    """

    model = load_model(cfg_synthesis['model_type'], cfg_synthesis['model_statedict_path'], cfg_synthesis['model_config_dict']).to(cfg_synthesis['device']) # Gets the model
    lean_sampler = get_lean_synthesis(cfg_synthesis['model_type'], model) # Gets the lean synthesizer

    synthesized_spectra = synthesize(cfg_synthesis, lean_sampler, ab_tensor) # Synthesizes the spectra using lean synthesizer

    return synthesized_spectra, ab_tensor
