import torch.nn.functional as F
import torch

def loss_mse_sam(x0: torch.tensor, x0_hat: torch.tensor, ratio: float):
    """
    Loss function to calculate the MSE + SAM from a reconstructed spectrum x0 (actual) and x0_hat (reconstructed). Assumes horiz inputs.

    Parameters
        - x0 (torch.tensor), Actual spectrum x0
        - x0_hat (torch.tensor), The reconstructed x0
        - Ratio (float), Ratio is a hyperparam between MSE and SAM

    Returns
        - Loss (float), MSE + Lambda * SAM
    """

    # The MSE loss, straightforward
    loss_mse = F.mse_loss(x0_hat, x0)

    # The SAM loss, compute cos similarity first (which is the actual SAM formula, without arccos)
    cos_sim = F.cosine_similarity(x0_hat, x0, dim=1)
    loss_sam = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))

    return loss_mse + ratio * loss_sam # Return the total loss