import torch.nn.functional as F
import torch.nn as nn
import torch

import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[1]
sys.path.append(str(project_root))
from defs.training_defs.loss import loss_mse_sam

class LossDDPM(nn.Module):
    """
    A very flexible loss the be used in our DDPM and above models.

    Args:
        noise_loss_type (str): Loss type for the predicted noise, options are 'l1', 'l2', 'huber'
        use_recons_loss (str): What to do with the spectral reconstruction, options are 'add_loss' (add to loss), 'observe' (returned, not used in tot_loss), 'none' (not returned, or added)
        ratio_noise_recons (float): Ratio to sum the recons and noise loss ratio is defined as recons/noise
    """
    def __init__(self, noise_loss_type:(str)='l2', use_recons_loss:(str)='observe', ratio_noise_recons:(float)=0.01):
        super().__init__()

        if noise_loss_type == 'l1': self.noise_criterion = nn.L1Loss()
        elif noise_loss_type == 'l2': self.noise_criterion = nn.MSELoss() 
        elif noise_loss_type == 'huber': self.noise_criterion = nn.HuberLoss()
        else: raise ValueError(f"Unknown/Unsupported noise_loss_type: {noise_loss_type}")

        if use_recons_loss in ['none', 'observe', 'add_loss']: 
            self.use_recons_loss = use_recons_loss
            if use_recons_loss == 'add_loss':
                self.ratio_noise_recons = ratio_noise_recons
        else: raise ValueError(f"Unknown/Unsupported use_recons_loss: {use_recons_loss}")

    def _recons_loss(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor)) -> torch.Tensor:

        # The SAM loss, compute cos similarity first (which is the actual SAM formula, without arccos)
        eps = 1e-7
        cos_sim = F.cosine_similarity(x_0_hat, x_0, dim=1)
        loss_sam = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)).mean() # Take the mean so that we get a scalar
                                                                                  # Clamp it with some eps, to prevent instability
 
        # The sam loss of spectral reconstruction
        return loss_sam
    
    def forward(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), x_n_hat:(torch.Tensor), x_n:(torch.Tensor)):

        """
        Gets the loss given some x_0 reconstruction, and predicted x_n

        Args:
            x_0_hat (torch.Tensor): Predicted clean spectra at t=0
            x_0 (torch.Tensor): Actual clean spectra at t=0
            x_n_hat (torch.Tensor): Predicted noise at temperature t
            x_n (torch.Tensor): Actual noise at temperature t

        Returns:
            total_loss (torch.Tensor): The total loss to be backpropagated
            noise_loss (torch.Tensor): The noise loss component
            recons_loss (torch.Tensor or None): The reconstruction loss component, or None if not used
        """

        if self.use_recons_loss == 'none':
            noise = self.noise_criterion(x_n_hat, x_n)
            return noise, noise, None
        
        elif self.use_recons_loss == 'observe':
            noise = self.noise_criterion(x_n_hat, x_n)
            recons = self._recons_loss(x_0_hat, x_0)
            return noise, noise, recons

        elif self.use_recons_loss == 'add_loss':
            noise = self.noise_criterion(x_n_hat, x_n)
            recons = self._recons_loss(x_0_hat, x_0)
            return noise + self.ratio_noise_recons * recons, noise, recons
        
    def sample_loss(self, x_0_hat, x_0):

        return self._recons_loss(x_0_hat, x_0)
    


class LossKLSAM(nn.Module):
    """
    Loss combining Kullback-Leibler, Spectral angle mapper and mean squared error loss used in VCCAE model
    """
    def __init__(self):
        super().__init__()

        self.loss_criterion = nn.MSELoss()
    
    def forward(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), mu:(torch.Tensor), variation:(torch.Tensor), 
                SAM_coefficient=1, KL_coefficient=1, MSE_coefficient=1, msesam_ratio = 0.05):
        """
        Gets the loss given some x_0 reconstruction, encoded mu and accounts for variation done by the algorithm

        Coefficients:
            a: SAM
            b: KL
            c: MSE
        """
        sam_criterion = loss_mse_sam(ratio=msesam_ratio)
        estimated = x_0_hat
        actual = x_0
        a = SAM_coefficient
        b = KL_coefficient
        c = MSE_coefficient

        # Loss
        pred_loss = nn.functional.mse_loss(estimated, actual[0])
        divergence = -0.5 * torch.sum(1 + variation - mu.pow(2) - variation.exp()) # Kullback-Leibler
        sam_loss = sam_criterion.calc(estimated, actual)
        return (a * sam_loss) + (b * divergence) + (c * pred_loss)