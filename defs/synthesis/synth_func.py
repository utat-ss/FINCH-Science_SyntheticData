"""
Here, we define different classes for the data synthesis operations.
"""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from .abundance_sampler import *

class SpectralSampler(ABC):
    def __init__(self, model:(nn.Module), lean:(bool)=True, abundance_sampler:(AbundanceSampler)=None):
        self.model = model
        self.lean = lean
        self.model.eval()

        if not self.lean:
            if abundance_sampler is None:
                raise ValueError(f"lean={self.lean} but abundance_sampler=None. Must provide an ab sampler if not using lean")
            self.abundance_sampler = abundance_sampler
            self._enforce_requirements_nonlean()
        else:
            self.abundance_sampler = None

    def _enforce_requirements_nonlean(self):
        if self.__class__.__call__ is SpectralSampler.__call__:
            raise NotImplementedError(f"'{self.__class__.__name__}' must have '__call__' implemented, when not lean")
        
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"'__call__' is not available when lean=True, either don't use this or override not implemented __call__ by lean=False")

    @abstractmethod
    def predefined_ab_sample(self, ab_tensor:(torch.Tensor)) -> torch.Tensor:
        pass

class GaussianDiffusionSampler(SpectralSampler):
    def __init__(self, model:(nn.Module), lean:(bool)=True, abundance_sampler:(AbundanceSampler)=None):
        super().__init__(model, lean, abundance_sampler)

    def __call__(self):
        if self.lean: # Keep the user from calling __call__ when lean mode is on
            super().__call__()

        # Gets a random abundance tensor from the abundance sampler
        ab_tensor = self.abundance_sampler()

        # Samples the data from the random abundance tensor
        with torch.no_grad():
            with self.model.use_ema():
                sampled_data, _ = self.model.sample(ab_tensor) # Throw away the actual noise patters, not needed

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu() # Returns it 

    def predefined_ab_sample(self, ab_tensor):
        
        with torch.no_grad():
            with self.model.use_ema():
                sampled_data, _ = self.model.sample(ab_tensor)

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu()
    
class CCVAESampler(SpectralSampler):

    def __init__(self, model:(nn.Module), lean:(bool)=True, abundance_sampler:(AbundanceSampler)=None, seed:(int)=3169):
        super().__init__(model, lean, abundance_sampler)
        self.noise_dim = None # Initialize this to none, it will be changed whether lean=True/False

        self.device = next(model.parameters()).device # Infer the device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed) # Get the generator

    def __call__(self):
        if self.lean: # If lean is enabled, prevent user from using __call__
            super().__call__()

        ab_tensor = self.abundance_sampler()
        noise_tensor = self._noise_sampler(ab_tensor)

        with torch.no_grad():
            latent_vector = self.model.conditioner(noise_tensor, ab_tensor)
            sampled_data = self.model.decoder(latent_vector)

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu()

    def _noise_sampler(self, ab_tensor:(torch.Tensor)) -> torch.Tensor:

        if self.noise_dim is None:
            self.noise_dim = self.model.latent_dim - ab_tensor.shape[1] # Gets (B, noise_dim = latent_dim - ab_dim)

        current_batch = ab_tensor.shape[0]
    
        return torch.rand(size=(current_batch, self.noise_dim), device=ab_tensor.device, generator=self.generator)
    
    def predefined_ab_sample(self, ab_tensor):
        
        noise_tensor = self._noise_sampler(ab_tensor)

        with torch.no_grad():
            latent_vector = self.model.conditioner(noise_tensor, ab_tensor)
            sampled_data = self.model.decoder(latent_vector)

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu()


class TCVAESampler(SpectralSampler):
    def __init__(self, model:(nn.Module), lean:(bool)=True, abundance_sampler:(AbundanceSampler)=None, seed:(int)=3169):
        super().__init__(model, lean, abundance_sampler)
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def __call__(self):
        if self.lean:
            super().__call__()

        ab_tensor = self.abundance_sampler()
        return self.predefined_ab_sample(ab_tensor)

    def predefined_ab_sample(self, ab_tensor:(torch.Tensor)):
        ab_tensor = ab_tensor.to(self.device)

        with torch.no_grad():
            sampled_data = self.model.sample_from_conditions(
                ab_tensor,
                generator=self.generator,
            )

        return sampled_data.detach().cpu(), ab_tensor.detach().cpu()
