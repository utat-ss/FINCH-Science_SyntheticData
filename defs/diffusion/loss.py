import torch.nn.functional as F
import torch.nn as nn
import torch

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
    
    def forward(self, x_0_hat, x_0, x_n_hat, x_n):

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


class loss_mse:

    def __init__(self):

        """
        Loss function to calculate the MSE between actual and predicted noise
        """

        pass

    def __call__(self, xn: torch.tensor, xn_hat: torch.tensor):
        """
        Calculates the loss

        Parameters
            - xn (torch.tensor), Actual noise that was added at temperature t.
            - xn_hat (torch.tensor), The predicted xn at temperature t.

        Returns
            - Loss (float), MSE
        """

        # The MSE loss, straightforward
        loss = F.mse_loss(xn_hat, xn) # This one already has in-built .mean() per se.

        return loss

class loss_mse_sam:

    def __init__(self, ratio: float = 0.01):

        """
        Loss function to calculate the MSE + SAM from a reconstructed spectrum x0 (actual) and x0_hat (reconstructed). Assumes horiz inputs.
            - Ratio (float), Ratio is a hyperparam between MSE and SAM. It is simply multiplied with the SAM loss.

        
        """
        self.ratio = ratio

    def epsilon_loss(self, xn: torch.tensor, xn_hat: torch.tensor):
        """
        Calculates the loss

        Parameters
            - xn (torch.tensor), Actual noise that was added at temperature t.
            - xn_hat (torch.tensor), The predicted xn at temperature t.

        Returns
            - Loss (float), MSE
        """

        # The MSE loss, straightforward
        loss_epsilon =  F.mse_loss(xn_hat, xn) # This one already has in-built .mean() per se.

        return loss_epsilon

    def recons_loss(self, x0: torch.tensor, x0_hat: torch.tensor):

        # The SAM loss, compute cos similarity first (which is the actual SAM formula, without arccos)
        cos_sim = F.cosine_similarity(x0_hat, x0, dim=1)
        loss_sam = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)).mean() # Take the mean so that we get a scalar
 
        # The sam loss of spectral reconstruction
        return loss_sam
    
    def __call__(self, x0: torch.tensor, x0_hat: torch.tensor, xn: torch.tensor, xn_hat: torch.tensor):

        """
        Given the actual and predicted x0 and xn, compute the final loss.

        Parameters
            - x0 (torch.tensor), Actual clean spectra at t=0
            - x0_hat (torch.tensor), Predicted clean spectra at t=0
            - xn (torch.tensor), Actual noise that was added at temperature t.
            - xn_hat (torch.tensor), The predicted xn at temperature t.
        
        Returns
            - loss (float), The final combined loss
        """

        epsilon_loss = self.epsilon_loss(xn, xn_hat)
        recons_loss = self.recons_loss(x0, x0_hat)

        total_loss = epsilon_loss + self.ratio * recons_loss

        return total_loss, epsilon_loss, recons_loss