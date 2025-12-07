import torch.nn.functional as F
import torch

class loss_mse_sam:
    def __init__(self, ratio: float = 0.01, use_logvar: bool = False, logvar_weight: float = 0.0, hybrid_lambda: float = 0.0):

        """
        Loss function to calculate the MSE + SAM from a reconstructed spectrum x0 (actual) and x0_hat (reconstructed). Assumes horiz inputs.
            - Ratio (float), Ratio is a hyperparam between MSE and SAM. It is simply multiplied with the SAM loss.

        
        """
        self.ratio = ratio
        self.use_logvar = use_logvar
        self.logvar_weight = logvar_weight  # weight multiplying the NLL if used
        self.hybrid_lambda = hybrid_lambda  # λ in Lhybrid = Lsimple + λ * Lvlb

    def epsilon_loss(self, xn: torch.Tensor, xn_hat: torch.Tensor, logvar: torch.Tensor = None):
        """
        Calculates the loss

        Parameters
            - xn (torch.tensor), Actual noise that was added at temperature t.
            - xn_hat (torch.tensor), The predicted xn at temperature t.

        Returns
            - Loss (float), MSE
        """

        # simple MSE
        mse = F.mse_loss(xn_hat, xn)
        if (logvar is None) or (not self.use_logvar):
            return mse
        # compute NLL term (per-dim), clamp logvar for stability
        logvar_clamped = torch.clamp(logvar, min=-20.0, max=20.0)
        var = torch.exp(logvar_clamped)
        nll = 0.5 * ((xn - xn_hat) ** 2) / (var + 1e-12) + 0.5 * logvar_clamped
        nll = nll.mean()
        if self.hybrid_lambda is not None and self.hybrid_lambda > 0.0:
            # hybrid: combine simple mse + λ * nll
            return mse + self.hybrid_lambda * nll
        # otherwise use weighted nll
        return nll * (1.0 + self.logvar_weight)

    def recons_loss(self, x0: torch.Tensor, x0_hat: torch.Tensor):

        # The SAM loss, compute cos similarity first (which is the actual SAM formula, without arccos)
        cos_sim = F.cosine_similarity(x0_hat, x0, dim=1)
        loss_sam = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)).mean()
        return loss_sam

    def __call__(self, x0: torch.Tensor, x0_hat: torch.Tensor, xn: torch.Tensor, xn_hat: torch.Tensor, logvar: torch.Tensor = None):
        
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
        
        epsilon_loss = self.epsilon_loss(xn=xn, xn_hat=xn_hat, logvar=logvar)
        recons_loss = self.recons_loss(x0=x0, x0_hat=x0_hat)
        total_loss = epsilon_loss + self.ratio * recons_loss
        return total_loss