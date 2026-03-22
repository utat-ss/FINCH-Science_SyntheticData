import torch.nn.functional as F
import torch.nn as nn
import torch

import sys
from pathlib import Path
project_root = Path.cwd().resolve().parents[1]
sys.path.append(str(project_root))


class Loss_KL_SAM_BAnneal(nn.Module):
    """
    Loss combining Kullback-Leibler, Spectral angle mapper and mean squared error loss used in VCCAE model, also uses Beta
    annealing.
    """
    def __init__(self, total_steps:(int), start_beta:(float)=0.0, target_beta:(float)=1.0, lambda_mse:(float)=1.0, lambda_kl:(float)=1.0, lambda_sam:(float)=1.0):
        super().__init__()

        # Beta annealing related
        self.total_steps = total_steps
        self.start_beta = start_beta
        self.beta = start_beta
        self.target_beta = target_beta
        self.current_step = 0

        self.eps = 1e-7 # Needed to adjust for cos

        # Coeffs
        self.lambda_mse = lambda_mse
        self.lambda_kl = lambda_kl
        self.lambda_sam = lambda_sam

    def step_beta(self):
        self.current_step += 1
        self.beta = self.start_beta + (self.target_beta - self.start_beta) * min(1.0, self.current_step / self.total_steps)
    
    def forward(self, pred:(torch.Tensor), actual:(torch.Tensor), mu:(torch.Tensor), logvar:(torch.Tensor)):
        """
        Gets the loss given some x_0 reconstruction, encoded mu and accounts for variation done by the algorithm

        Loss = lambda_mse * MSE + lambda_kl * beta * KL + lambda_sam * SAM
        """
        # Loss
        loss_MSE = F.mse_loss(pred, actual, reduction='none').sum(dim=1).mean()
        loss_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss_SAM = torch.acos(torch.clamp(F.cosine_similarity(pred, actual, dim=1), -1.0 + self.eps, 1.0 - self.eps)).mean()

        return self.lambda_mse*loss_MSE + self.lambda_kl*self.beta*loss_KL + self.lambda_sam*loss_SAM, loss_MSE, loss_KL, loss_SAM, self.beta
