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

    def _math_logic(self, tensor):
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)
        
    def __call__(self):
        ab = torch.rand(size=self.target_shape, generator=self.generator, device=self.device) # Generate the abundances 
        return self._math_logic(ab)
    
class NormAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes normal distribution as the init randomness.
    """
    def __init__(self, target_shape, seed, device):
        super().__init__(target_shape, seed, device)

    def _math_logic(self, tensor):
        tensor.abs_()
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)

    def __call__(self):
        ab = torch.randn(size=self.target_shape, generator=self.generator, device=self.device) # Generate the abundances 
        return self._math_logic(ab)
    
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
        
    def _math_logic(self, tensor):
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)
        
    def __call__(self):
        ab = torch.empty(size=self.target_shape, device=self.device)
        for i, alpha in enumerate(self.alphas): ab[:,i].gamma_(concentration=alpha, concentration_1=1.0, generator=self.generator)
        return self._math_logic(ab)

class LogNormalAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes log-normal distribtuions
    """
    def __init__(self, target_shape, seed, device, weights, mus, log_vars, alr_bool):
        super().__init__(target_shape, seed, device)

        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.mus = torch.tensor(mus, device=device, dtype=torch.float32)
        self.log_vars = torch.tensor(log_vars, device=device, dtype=torch.float32)

        self.scales = F.softplus(self.log_vars) + 1e-5

        self.K = self.weights.shape[0]
        self.dim = self.mus.shape[1]

        self.alr_bool = alr_bool

    def _sampling_logic(self):
        if self.alr_bool:
            cat = torch.distributions.Categorical(self.weights)
            ks = cat.sample((self.target_shape[0],))  # (Batch,)

            # Sample in alr
            eps = torch.randn(self.target_shape[0], self.dim, device=self.device)
            ab = self.mus[ks] + self.scales[ks] * eps  # (Batch, n_endmember - 1)

            # Get back to simplex
            ab = self._inv_alr(ab) # (Batch, n_endmember)
            return ab
        else:
            # Weird sampling
            randn = torch.randn(size=self.target_shape, device=self.device)
            ab = torch.exp(self.mus + self.sigmas * randn)
            return ab 

    def _inv_alr(self, ab):
        """
        Inverses the ALR.

        Args:
            ab_alr: Abundances on R^2, (..., 2)
        Returns:
            ab_simplex: Abundances on simplex, (..., 3)
        """
        expz = torch.exp(ab)
        denom = 1 + expz.sum(dim=-1, keepdim=True)
        x12 = expz / denom
        x3 = 1 / denom
        return torch.cat([x12, x3], dim=-1)

    def _math_logic(self, tensor):
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)

    def __call__(self):
        ab = self._sampling_logic()
        return self._math_logic(ab)

