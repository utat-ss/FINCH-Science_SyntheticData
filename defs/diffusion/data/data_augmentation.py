import torch.nn as nn
import torch

class SpectralAugmentation(nn.Module):
    """
    Data augmenter for the DDPM models. We need this because our ground truth dataset is insanely low in size.
    Assumes the normalization of the dataset in some capacity.

    Args:
        seed (int): Seed of the randomizers
        jitter_scale (float): The scale for the randomized jitter
        amp_scale (float): The scale for random amplitude multiplication the [1, 1-amp_scale] is randomly chosen and uniformly multiplied
    """
    def __init__(self, seed:(int)=3169, jitter_scale:(float)=1e-4, amp_scale:(float)=1e-2):
        super().__init__()
        self.generator = torch.Generator(); self.generator.manual_seed(seed)
        self.jitter_scale = jitter_scale
        self.amp_scale = amp_scale 
    
    def forward(self, x_0:(torch.Tensor)) -> torch.Tensor:
        """
        Takes in a ground truth data, randomly augments it.

        Logic:
            y = x_0 * scale + noise * jitter_scale

        Args:
            x_0 (torch.Tensor): The data from ground truth dataset
        
        Returns:
            x_aug (torch.Tensor): The augmented data
        """
        # Amplitude perturbation, works as a contrast jitter, i.e. jitter applied throughout
        rand_val = torch.rand(x_0.shape[0], 1, generator=self.generator, dtype=x_0.dtype, device='cpu').to(device=x_0.device)

        scale = (rand_val * self.amp_scale * 2) + (1 - self.amp_scale)
        # For amp_scale = 0.01, gets the scale to be between 0.99-1.01

        # Random jitter
        noise = torch.randn(size=x_0.shape, generator=self.generator, dtype=x_0.dtype, device='cpu').to(device=x_0.device)
        # This creates a random jitter of the same size, with mean 0 and stdev 1

        return x_0 * scale + noise * self.jitter_scale

