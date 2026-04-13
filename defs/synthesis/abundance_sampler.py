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

class LogNormalAbSamp(AbundanceSampler):
    """
    An abundance sampler that assumes log-normal distribution.
    """
    def __init__(self, target_shape, seed, device, mix_weights:(list[float]), mus:(list[list[float]]), Ls:(list[list[list[float]]]), zero_thresh:(float)=0.0001):
        super().__init__(target_shape, seed, device)

        # Store the inputs
        self.mix_weights = torch.tensor(mix_weights, dtype=torch.float32, device=device)
        self.mus = torch.tensor(mus, dtype=torch.float32, device=device)

        # Permanent params, will change this majorly later on
        _Ls = torch.tensor(Ls, dtype=torch.float32, device=device)

        # Validating shapes, important we do this sanity check
        K = len(mix_weights)
        n_endmembers = target_shape[1]
        assert self.mus.shape == (K, n_endmembers), f"Mus shape ({self.mus.shape}) must match the expected shape of mixtures and endmembers ({(K, n_endmembers)})"

        # Get the scale trils from Ls
        L_tril = torch.tril(_Ls)
        diag = torch.diagonal(L_tril, dim1=-2, dim2=-1)
        off_diag = L_tril - torch.diag_embed(diag)
        pos_diag = F.softplus(diag) + self.eps
        self.scale_trils = off_diag + torch.diag_embed(pos_diag)   

        # For the thresholding during sampling
        self.zero_thresh = zero_thresh

    def math_logic(self, tensor):
        """
        First, normalizes as a sanity check, then clips the values below the thresh to 0, then normalizes again
        We have to do this because by the math that takes place, we can mathematically never have abundances 
        Really close to 0
        """
        tensor = F.normalize(tensor, p=1, dim=1, eps=self.eps)
        mask = tensor < self.zero_thresh
        tensor = tensor.masked_fill(mask, 0.0)
        return F.normalize(tensor, p=1, dim=1, eps=self.eps)
    
    def __call__(self):
        """
        Randomly creates the abundances using the learned distribution
        """
        batch_size, n_endmembers = self.target_shape[0], self.target_shape[1]

        # We have to randomly choose one of the learned distributions, for each of the batch entries
        selected_components = torch.multinomial( self.mix_weights, batch_size, replacement=True, generator=self.generator) # Size is (Batch,)
        selected_mus = self.mus[selected_components] # Shape is (Batch, n_endmembers)
        selected_trils = self.scale_trils[selected_components] # Shape is (Batch, n_endmembers, n_endmembers)

        # Sample a normal dist
        epsilon = torch.randn(batch_size, n_endmembers, 1, device=self.device, generator=self.generator)

        # Perform the operation to get multivar normal in log-space:
        # (Batch, n_endmembers, 1) = (Batch, n_endmembers, 1) + (Batch, n_endmembers, n_endmembers) @ (Batch, n_endmembers, 1)
        log_ab = selected_mus + (selected_trils @ epsilon).squeeze(-1)
        sampled_ab = torch.exp(log_ab)

        return self.math_logic(sampled_ab)
