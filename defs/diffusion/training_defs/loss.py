import torch.nn.functional as F
import torch

class loss_mse_sam:

    def __init__(self, ratio: float = 0.01):

        """
        Loss function to calculate the MSE + SAM from a reconstructed spectrum x0 (actual) and x0_hat (reconstructed). Assumes horiz inputs.
            - Ratio (float), Ratio is a hyperparam between MSE and SAM. It is simply multiplied with the SAM loss.

        
        """
        self.ratio = ratio

    def epsilon_loss(xn: torch.tensor, xn_hat: torch.tensor):
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

    def recons_loss(x0: torch.tensor, x0_hat: torch.tensor):

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

        epsilon_loss = self.epsilon_loss(xn=xn, xn_hat=xn_hat)
        recons_loss = self.recons_loss(x0=x0, x0_hat=x0_hat)

        total_loss = epsilon_loss + self.ratio * recons_loss

        return total_loss