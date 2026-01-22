"""
This module contains classes related to the structuring of synthesized data
"""

import pandas as pd
import torch
import numpy as np

class SpectralDataCollector:

    def __init__(self, naming_base:(str), counter:(int)=0, spectral_range:(list[int])=[400, 2490]):

        self.naming_base = naming_base
        self.counter = counter
        self.names_list = []
        self.abundances_list = []
        self.spectra_list = []
        self.spectral_range = spectral_range
        self.wavelengths = np.arange(spectral_range[0], spectral_range[1]+1, 10)

    def add_data(self, abundances:(torch.Tensor), spectra:(torch.Tensor)):

        # Append the data to the lists
        self.abundances_list.append(abundances)
        self.spectra_list.append(spectra)

        # Create the names
        number_range = range(self.counter, self.counter + abundances.shape[0])
        for i in number_range:
            self.names_list.append(f"{self.naming_base}_{i}")
        self.counter += abundances.shape[0] # Move the counter with batch size

    def to_dataframe(self) -> pd.DataFrame:
        # Concatenate the data
        all_abundances = torch.cat(self.abundances_list, dim=0).cpu().numpy()
        all_spectra = torch.cat(self.spectra_list, dim=0).cpu().numpy()

        # Create the dataframe
        df_abundances = pd.DataFrame(all_abundances, columns=['gv_fraction', 'npv_fraction', 'soil_fraction'])
        df_spectra = pd.DataFrame(all_spectra, columns=self.wavelengths)
        df_names = pd.DataFrame(self.names_list, columns=["Spectra"])
        df_use = pd.DataFrame(['synthesized'] * len(self.names_list), columns=["use"]) # Use column
        df_rwc = pd.DataFrame("", index=range(len(self.names_list)), columns=["RWC index", "Calculated RWC"]) # Empty RWC columns

        df_final = pd.concat([df_names, df_abundances, df_rwc, df_use, df_spectra], axis=1)

        return df_final
    
    def save_to_csv(self, path:(str)):
        df_final = self.to_dataframe()
        df_final.to_csv(path, index=False)


