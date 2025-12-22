"""
Here, we define different classes for the data synthesis operations.
"""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from .abundance_sampler import *

class SpectralSampler(ABC):

    def __init__(self, model:(nn.Module), abundance_sampler:(AbundanceSampler)):

        self.model = model
        self.abundance_sampler = abundance_sampler

        self.model.eval() # Set the model in evaluation mode to make many norm layers reproducible

    @abstractmethod
    def __call__(self):
        pass

class GaussianDiffusionSampler(SpectralSampler):

    def __init__(self, model:(nn.Module), abundance_sampler:(AbundanceSampler)):
        super().__init__(model, abundance_sampler)

    def __call__(self):

        # Gets a random abundance tensor from the abundance sampler
        ab_tensor = self.abundance_sampler()

        # Samples the data from the random abundance tensor
        with self.model.use_ema():
            sampled_data, _ = self.model.sample(ab_tensor) # Throw away the actual noise patters, not needed

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu() # Returns it 
    
class AutoEncoderSampler(SpectralSampler):

    def __init__(self, model:(nn.Module), abundance_sampler:(AbundanceSampler)):
        super().__init__(model, abundance_sampler)

        self.target_shape = self.abundance_sampler.target_shape
        self.generator = self.abundance_sampler.generator

    def __call__(self):

        noise_tensor = self._noise_sampler()
        ab_tensor = self.abundance_sampler()

        with torch.no_grad():
            latent_vector = self.model.conditioner(noise_tensor, ab_tensor)
            sampled_data = self.model.decoder(latent_vector)

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu()

    def _noise_sampler(self) -> torch.Tensor:
        return torch.rand(size=self.target_shape, generator=self.generator)

