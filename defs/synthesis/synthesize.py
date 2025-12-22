"""
Here, we define the different function that will be used during the synthesis process itself.

These functions are to mainly handle the synthesized data, and save them into a file
"""

from .data import SpectralDataCollector
import logging

def synthesize_data(cfg_synthesis:(dict), cfg_export:(dict), unnormalizer, synthesizer):

    # Get the number of samples to generate
    n_targets = cfg_synthesis['n_targets']
    device = cfg_synthesis['device']
    
    naming_base = cfg_export['naming_base']
    spectral_range = cfg_export['spectral_range']

    # Setup the save path
    master_path = cfg_export['master_path']
    ksi_train_path = cfg_export['ksi_train_path']
    if not ksi_train_path.endswith('.csv'): ksi_train_path += '.csv'
    ksi_train_path = master_path + ksi_train_path

    # Initialize the data collector
    data = SpectralDataCollector(naming_base=naming_base, spectral_range=spectral_range)

    for _ in range(n_targets):
        logging.info(f"Synthesizing batch {_+1} out of {n_targets}")
        spectra, abundances = synthesizer() # Get the synthesized data and abundances
        spectra = unnormalizer(spectra) # Unnormalize the data
        data.add_data(abundances, spectra)

    # Save the data to CSV
    logging.info(f"Saving synthesized data to {ksi_train_path}")
    data.save_to_csv(ksi_train_path)
    logging.info("Data successfully saved.")
