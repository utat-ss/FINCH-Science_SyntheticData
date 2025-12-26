import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Tuple, Optional, Callable

class LossGaussianDiffusion(nn.Module):
    """
    A very flexible loss the be used in our Gaussian Diffusion models.

    Args:
        noise_loss_type (str): Loss type for the predicted noise, options are 'l1', 'l2', 'huber'
        use_recons_loss (str): What to do with the spectral reconstruction, options are 'add_loss' (add to loss), 'observe' (returned, not used in tot_loss), 'none' (return None, don't add)
        use_fft_loss (str): What to do with the fft comparison, options are 'add_loss' (add to loss), 'observe' (returned, not used), 'none' (return None, don't add)
        use_tv_loss (loss): What to do with the total value, options are 'add_loss' (add to loss), 'observe' (returned, not used), 'none' (return None, don't add)
        ratio_recons_loss (float): Ratio to multiply the reconstruction loss with
        ratio_fft_loss (float): Ratio to multiply the reconstruction loss with
        ratio_tv_loss (float): Ratio to multiply the total value loss with
    """
    def __init__(
            self,
            noise_loss_type:(str)='l2',
            use_recons_loss:(str)='add_loss',
            use_fft_loss:(str)='add_loss',
            use_tv_loss:(str)='add_loss',
            ratio_recons_loss:(float)=0.5,
            ratio_fft_loss:(float)=1.0,
            ratio_tv_loss:(float)=1.0
    ):
        super().__init__()

        if noise_loss_type == 'l1': self.noise_criterion = nn.L1Loss()
        elif noise_loss_type == 'l2': self.noise_criterion = nn.MSELoss() 
        elif noise_loss_type == 'huber': self.noise_criterion = nn.HuberLoss()
        else: raise ValueError(f"Unknown/Unsupported noise_loss_type: {noise_loss_type}")

        if use_recons_loss in ['add_loss', 'observe', 'none']: 
            self.use_recons_loss = use_recons_loss
            if use_recons_loss == 'add_loss':
                self.ratio_recons_loss = ratio_recons_loss
        else: raise ValueError(f"Unknown/Unsupported use_recons_loss: {use_recons_loss}")

        if use_fft_loss in ['add_loss', 'observe', 'none']:
            self.use_fft_loss = use_fft_loss
            if use_fft_loss == 'add_loss':
                self.ratio_fft_loss = ratio_fft_loss
        else: raise ValueError(f"Unknown/Unsupported use_fft_loss: {use_fft_loss}")
        self.fft_loss_component = nn.L1Loss()

        if use_tv_loss in ['add_loss', 'observe', 'none']:
            self.use_tv_loss = use_tv_loss
            if use_tv_loss == 'add_loss':
                self.ratio_tv_loss = ratio_tv_loss
        else: raise ValueError(f"Unknown/Unsupported use_tv_loss: {use_tv_loss}")

    def _recons_loss(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), unnorm_func:(Optional[Callable[[torch.Tensor], torch.Tensor]])=None) -> torch.Tensor:
        """
        This loss calculates the spectral angle between the reconstructed and true spectra, to get a rough measure
        of how similary they are spectrally.

        Args:
            x_0_hat (torch.Tensor): The reconstructed spectra at t=0, outputted from the Gaussian Diffusion model
            x_0 (torch.Tensor): The true spectra at t=0
            unnorm_func (Lambda): A lambda function of spectral unnormalizer

        Returns:
            loss_sam (torch.Tensor): A spectral angle between the true spectrum vs reconstructed spectrum at t=0

        Logic:
            A := x_0_hat, B := x_0
            cos_sim = dot(A,B)/Norm(A)*Norm(B)
            loss_sam = arccos(cos_sim)
        """
        # Unnormalize the passed in spectra first, if not None
        if unnorm_func is not None:
            x_0_hat = unnorm_func(x_0_hat)
            x_0 = unnorm_func(x_0)

        # The SAM loss, compute cos similarity first (which is the actual SAM formula, without arccos)
        eps = 1e-7
        cos_sim = F.cosine_similarity(x_0_hat, x_0, dim=1)
        loss_sam = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)).mean() # Take the mean so that we get a scalar
                                                                                  # Clamp it with some eps, to prevent instability
 
        # The sam loss of spectral reconstruction
        return loss_sam
    
    def _fft_loss(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor)) -> torch.Tensor:
        """
        This loss calculates the L1 difference in the FT space, allowing us to dynamically filter different sources of signal
        such as noise, this allows us to preserve the specific features of spectra, by not defining them as noise. Acts as a low
        pass filter. Assuming the user uses statistical norming, which by Parseval's theorem, does not require unnorming. But if
        using other forms of non-linear norming, then the data must be unnormed

        Args:
            x_0_hat (torch.Tensor): The reconstructed spectra at t=0, outputted from the Gaussian Diffusion model
            x_0 (torch.Tensor): The true spectra at t=0

        Returns:
            loss_fft (torch.Tensor): The L1 difference of FFT signals of both reconstructed and true spectra

        Logic:
            A := FT(x_0_hat); B := FT(x_0)
            A_abs = abs(A); B_abs = abs(B)
            loss_fft = L1(A_abs, B_abs)
        """
        # Applies the fast fourier transform along the last dimension in real domain, since it is (Batch, n_bands)
        # We don't have an in place option for this, :( the pytorch team did not think it was cool enough
        x_0_hat = torch.fft.rfft(x_0_hat, dim=-1)
        x_0 = torch.fft.rfft(x_0, dim=-1)

        # Takes the absolute value, since we don't need the phase information
        x_0_hat = torch.abs(x_0_hat)
        x_0 = torch.abs(x_0)

        # The FFT loss component, L1 loss between the absolute FFT values
        loss_fft = self.fft_loss_component(x_0_hat, x_0)

        # Returns the FFT loss
        return loss_fft
    
    def _tv_loss(self, x_0_hat:(torch.Tensor)) -> torch.Tensor:
        """
        Total variation loss, encourages smoothness in the reconstructed spectra. It is simply how much variation we have
        in each step (x_{i+1} - x_{i}). We then average out the absolute value of these differences. User must be incredibly
        careful about this loss since it is explicitly enforec on x_0_hat, therefore, as our predictions get better, it may
        start acting counterproductive and smoothening the spectra. Acts as a high-pass filter. Theory is that if we have a
        lot of smaller differences, that loss will be much bigger than a couple of large spectra feature differences (to prove
        this for yourself, just take derivatives of different sinusoidal signals).

        Args:
            x_0_hat (torch.Tensor): The reconstructed spectra at t=0, outputted from the Gaussian Diffusion model

        Returns:
            loss_tv (torch.Tensor): The total variation of the reconstructed spectrum
        
        Logic:
            shift := x_{i+1} - x{i} for all 0<i<n_bands
            variation := Abs(shift)
            loss_tv = 1/(n_bands-1) * Sum(variation)
        """
        # They are norm invariant for linear norm, so, we'll save the compute assuming the normer is linear
        # Shift and subtract the two tensors
        shift = x_0_hat[..., 1:] - x_0_hat[..., :-1]
        # Take the abs, not taking in-place cuz I don't feel like it
        variation = torch.abs(shift)
        # Take the average of the variation
        loss_tv = torch.mean(variation, dim=-1) # After this, (Batch,)
        loss_tv = torch.mean(loss_tv) # After this, (int)

        # Returns the Total Variation loss
        return loss_tv
    
    def forward(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), x_n_hat:(torch.Tensor), x_n:(torch.Tensor), unnorm_func:(Optional[Callable[[torch.Tensor], torch.Tensor]])=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Gets the training loss given some x_0_hat (reconstruction), x_0, x_n_hat (predicted), and x_n

        Args:
            x_0_hat (torch.Tensor): Predicted clean spectra at t=0
            x_0 (torch.Tensor): Actual clean spectra at t=0
            x_n_hat (torch.Tensor): Predicted noise at temperature t
            x_n (torch.Tensor): Actual noise at temperature t
            unnorm_func (Lambda): The lambda that acts as unnormalizer of data

        Returns:
            total (torch.Tensor): The total loss to be backpropagated
            noise (torch.Tensor): The noise loss component
            recons (torch.Tensor or None): The reconstruction loss component, or None if not used
            fft (torch.Tensor or None): The fft loss component, or None if not used
            tv (torch.Tensor or None): The tv loss component, or None if not used

        Logic:
            total = noise + ratio_recons * recons + ratio_fft * fft + ratio_tv * tv
        """
        # Section for noise_criterion
        noise = self.noise_criterion(x_n_hat, x_n)
        total = noise

        # Section for reconstruction loss
        if self.use_recons_loss == 'add_loss':
            recons = self._recons_loss(x_0_hat, x_0, unnorm_func)
            total += self.ratio_recons_loss * recons
        elif self.use_recons_loss == 'observe':
            with torch.no_grad(): recons = self._recons_loss(x_0_hat, x_0, unnorm_func)
        else: recons = None

        # Section for fft loss
        if self.use_fft_loss == 'add_loss':
            fft = self._fft_loss(x_0_hat, x_0)
            total += self.ratio_fft_loss * fft
        elif self.use_fft_loss == 'observe':
            with torch.no_grad(): fft = self._fft_loss(x_0_hat, x_0)
        else: fft = None

        # Section for tv loss
        if self.use_tv_loss == 'add_loss':
            tv = self._tv_loss(x_0_hat)
            total += self.ratio_tv_loss * tv
        elif self.use_tv_loss == 'observe':
            with torch.no_grad(): tv = self._tv_loss(x_0_hat)
        else: tv = None

        return total, noise, recons, fft, tv

    def sample_loss(self, x_0_hat:(torch.Tensor), x_0:(torch.Tensor), unnorm_func:(Optional[Callable[[torch.Tensor], torch.Tensor]])=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Gets the sampling loss given some x_0_hat (reconstruction) and x_0

        Args:
            x_0_hat (torch.Tensor): Predicted clean spectra at t=0
            x_0 (torch.Tensor): Actual clean spectra at t=0
            unnorm_func (Lambda): The lambda that acts as unnormalizer of data

        Returns:
            total (torch.Tensor or None): The total sampling loss
            recons (torch.Tensor or None): The reconstruction loss component, or None if not used
            fft (torch.Tensor or None): The fft loss component, or None if not used
            tv (torch.Tensor or None): The tv loss component, or None if not used

        Logic:
            total = ratio_recons * recons + ratio_fft * fft + ratio_tv * tv
        """

        total = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)

        if self.use_recons_loss == 'add_loss':
            with torch.no_grad(): recons = self._recons_loss(x_0_hat, x_0, unnorm_func)
            total += self.ratio_recons_loss * recons
        elif self.use_recons_loss == 'observe':
            with torch.no_grad(): recons = self._recons_loss(x_0_hat, x_0, unnorm_func)
        else: recons = None

        if self.use_fft_loss == 'add_loss':
            with torch.no_grad(): fft = self._fft_loss(x_0_hat, x_0)
            total += self.ratio_fft_loss * fft
        elif self.use_fft_loss == 'observe':
            with torch.no_grad(): fft = self._fft_loss(x_0_hat, x_0)
        else: fft = None

        if self.use_tv_loss == 'add_loss':
            with torch.no_grad(): tv = self._tv_loss(x_0_hat)
            total += self.ratio_tv_loss * tv
        elif self.use_tv_loss == 'observe':
            with torch.no_grad(): tv = self._tv_loss(x_0_hat)
        else: tv = None

        if total.item() == 0.0: total = None

        return total, recons, fft, tv
