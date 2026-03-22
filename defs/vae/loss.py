import torch.nn.functional as F
import torch.nn as nn
import torch

import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[1]
sys.path.append(str(project_root))


class LossKLSAM(nn.Module):
    """
    Loss combining Kullback-Leibler, Spectral angle mapper and mean squared error loss used in VCCAE model
    """
    def __init__(self):
        super().__init__()

        self.loss_criterion = nn.MSELoss()
        self.eps = 1e-7 # Needed to adjust for cos
    
    def forward(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), mu:(torch.Tensor), variation:(torch.Tensor), 
                SAM_coefficient=0.5, KL_coefficient=0.1, MSE_coefficient=1, msesam_ratio = 0.05):
        """
        Gets the loss given some x_0 reconstruction, encoded mu and accounts for variation done by the algorithm

        Coefficients:
            a: SAM
            b: KL
            c: MSE
        """
        estimated = x_0_hat
        actual = x_0
        a = SAM_coefficient
        b = KL_coefficient
        c = MSE_coefficient
        d = 1

        # Loss
        pred_loss = nn.functional.mse_loss(estimated, actual)
        divergence = -0.5 * torch.sum(1 + variation - mu.pow(2) - variation.exp()) # Kullback-Leibler
        c1 = F.cosine_similarity(estimated, actual, dim=1)
        sam_loss = d * torch.acos(torch.clamp(c1, -1.0 + self.eps, 1.0 - self.eps)).mean() # Take the mean so that we get a scalar

        loss = (a * sam_loss) + (b * divergence) + (c * pred_loss)
        return loss