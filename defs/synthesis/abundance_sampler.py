from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

class AbundanceSampler(ABC):
    """
    This is the abstract class for the abundance samplers.

    Args:
        taret_shape (tuple): The tuple of the wanted spectra, usually (Batch, n_endmembers)
        seed (int): Seed for the generator
        device (torch.device): The device to export the abundance tensor in
    """
    def __init__(self, target_shape:(tuple), seed:(int), device:(torch.device)):
        
        self.target_shape = target_shape
        self.device = device

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        self.eps = 1e-8

    @abstractmethod
    def __call__(self):
        pass

class UniAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes some uniform randomness.
    """
    def __init__(self, target_shape, seed, device):
        super().__init__(target_shape, seed, device)

    def math_logic(self, tensor):
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)
        
    def __call__(self):
        ab = torch.rand(size=self.target_shape, generator=self.generator, device=self.device) # Generate the abundances 
        return self.math_logic(ab)
    
class NormAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes normal distribution as the init randomness.
    """
    def __init__(self, target_shape, seed, device):
        super().__init__(target_shape, seed, device)

    def math_logic(self, tensor):
        tensor.abs_()
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)

    def __call__(self):
        ab = torch.randn(size=self.target_shape, generator=self.generator, device=self.device) # Generate the abundances 
        return self.math_logic(ab)
    
class DirAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes dirichlet distribution as the init randomness.
    """
    def __init__(self, target_shape, seed, device, alpha:tuple|list|int|float):
        super().__init__(target_shape, seed, device)

        n_endmembers = target_shape[1]

        if isinstance(alpha, (float, int)):
            self.alphas = (float(alpha),) * n_endmembers
        elif isinstance(alpha, (list, tuple)):
            assert len(alpha) == n_endmembers, f"Target n_endmembers: {n_endmembers} must be equal to length of alphas: {alpha}"
            self.alphas = alpha
        else:
            raise ValueError(f"Unknown/Unsupported alpha input type: {type(alpha)}")
        
    def math_logic(self, tensor):
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)
        
    def __call__(self):
        ab = torch.empty(size=self.target_shape, device=self.device)
        for i, alpha in enumerate(self.alphas): ab[:,i].gamma_(concentration=alpha, concentration_1=1.0, generator=self.generator)
        return self.math_logic(ab)

