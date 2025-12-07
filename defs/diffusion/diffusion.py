"""
Define the diffusion class here
"""

import torch
import torch.nn as nn
from .noise.noise_scheduling import *
from .noise.noise_sampling import *

class DDPM(nn.Module):

    def __init__(self, epsilon:(nn.Module), augmenter:(nn.Module), scheduler:(Schedule), t_sampler:(Sampling)):
        super().__init__()

        self.epsilon = epsilon # Take in the epsilon network
        self.augmenter = augmenter # Take in the data augmenter
        self.scheduler = scheduler # Take in the noise scheduler
        self.t_sampler = t_sampler # Take in the temp scheduler

        # Precompute all the scheduler params
        steps = self.scheduler.steps # Get the total amounts of steps from the scheduler
        t_s = torch.arange(1, steps + 1); self.register_buffer('t_s', t_s) # Time steps from 1 to T
        self.register_buffer('t_s_reverse', t_s.flip(0)) # Get the time steps in reverse order, T to 1
        self.register_buffer('alphas', self.scheduler.alpha_t(torch.cat((torch.tensor([0]), t_s)))) # Precompute all the alphas for each time step in order add t=0
        self.register_buffer('betas', self.scheduler.beta_t(torch.cat((torch.tensor([0]), t_s)))) # Precompute all the betas for each time step in order add t=0
        self.register_buffer('beta_tildas', self.scheduler.beta_tilda_t(t_s)) # Precompute all the beta tildas for each time step in order

        # Get the bands from the epsilon
        self.n_bands = self.epsilon.n_bands # Get the bands, used to randomly generate noise

    def _scheduled_call(self, x_0):

        """
        Randomly samples some noised data given some initial x_0, with given scheduler
        """

        # Take in some temperature matrix using the temperature sampler
        t = self.t_sampler(x_0) # Give the sampler the x_0 size, and get a t matrix size of (Batch,1)
        noise, x_t = self.scheduler.add_noise(x_0, t)

        # Return both the random time and the noised data related to it
        return t, noise, x_t

    def _recover_signal(self, x_t, t, ab):

        """
        Using the epsilon network, abundance condition, and the random noise, recover the actual signal x_0.

        Here, we simply rearrange the noise forward process to have:

        x_0 = (1/sqrt(alpha_bar_t)) * (x_t - sqrt(1 - alpha_bar_t) * noise, or epsilon)
        
        So, we simply use epsilon to predict how much noise was there on the signal that we got. And the rest is just algebra...
        """

        eps_pred = self.epsilon(x_t, t.to(torch.float32), ab) # Must convert the time to float
        alpha_bar_t = self.scheduler.gather(self.scheduler.alpha_bars, t, x_t.ndim).to(x_t.device)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        return x0_pred, eps_pred
    
    def training_procedure(self, x_0, ab):
        """
        Given some batched x_0 and ab (abundances related), it returns the predicted x_0. 
        Such predicted x_0 come from an entire procedure of forward noising and denoising using epsilon.
        The returned values will be used to get a loss to then backprop on epsilon (noise prediction) network.
        """
        x_0 = self.augmenter(x_0) # Augment the inputted data
        t, noise, x_t = self._scheduled_call(x_0) # Add some noise to it randomly
        x_0_hat, eps_pred = self._recover_signal(x_t, t, ab) # Recover the signal and pred the noise

        return x_0_hat, x_0, eps_pred, noise

    def _sample_step(self, x_t, t, ab):
        """
        A sampling substep where the signal is moved from x_t to x_(t-1).

        Args:
            x_t (torch.Tensor): Signal at time t
            t (int): Integer of time, temperature
            ab (torch.Tensor): Abundances tensor, condition for the epsilon

        Returns:
            x_(t-1): Signal at time t-1, noise of time t removed
        """

        t_tensor = torch.full((x_t.size(0),1), t, dtype=torch.long, device=x_t.device) # Create an empty tensor of the timesteps, size of (B,1)

        # Predict the noise that was added last step
        eps = self.epsilon(x_t, t_tensor.to(torch.float32), ab)
        if eps.ndim==3: eps=eps.squeeze(1) # Squeeze the channel dim of our eps, if we are getting (B, ch, n_bands) as out from it

        # Compute variance (sigma^2) / noise for stochastic sampling
        # For simplicity, use sqrt(beta_tilda_t) for this. The DDPM paper: https://arxiv.org/pdf/2006.11239
        sigma = torch.sqrt(self.scheduler.gather(self.beta_tildas, t_tensor-1, x_t.ndim)).to(torch.float32) # -1 on the time tensor because the 0th index of beta is actually beta tilda at t=1

        # Sample noise, without any noise at step t=0
        z = torch.randn_like(x_t) if t > 1 else 0.0

        # Gather alpha_t
        alpha_t = self.scheduler.gather(self.alphas, t_tensor, x_t.ndim).to(torch.float32)

        # Gather alpha_bar_t
        alpha_bar_t = self.scheduler.gather(self.scheduler.alpha_bars, t_tensor, x_t.ndim).to(torch.float32)

        # Update x_t for the next iter
        x_t = 1/torch.sqrt(alpha_t) * (x_t - ((1-alpha_t)/torch.sqrt(1-alpha_bar_t) * eps)) + sigma * z  

        return x_t

    def sample(self, ab, x_T=None):
        """
        Sample a signal using the diffusion model.

        Args:
            ab (Tensor); Abundance condition, shape [B, n_ab]
            x_T (Tensor, optional); Starting noisy signal. If none, Gaussian noise is used.

        Returns:
            x_0: (Tensor); Generated spectra, shape [B, n_bands]
            x_T: (Tensor); The high temperature spectrum before denoising, shape [B, n_bands]
        """
        device = ab.device # Get the device using ab 
        B = ab.size(0) # Get the batch number using ab

        if x_T is None: # If no x_T is given, define some using gaussian
            x_T = torch.randn(B, self.n_bands, device=device)
        
        # Reverse diffusion loop, for more detail on DDPM, check the paper on it.
        x_t = x_T # With such redefinition, we make sure that we preserve high temperature spectrum.

        for t in self.t_s_reverse: # Reversing the range so that: t = T, T-1, T-2, ..., 2, 1

            x_t = self._sample_step(x_t, t, ab)

        return x_t, x_T
