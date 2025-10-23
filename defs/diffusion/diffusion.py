"""
Define the diffusion class here
"""

import torch
import torch.nn as nn
import numpy as np

class cond_diffusion(nn.Module):

    def __init__(self, epsilon, scheduler):
        super().__init__()


        self.epsilon = epsilon # Take in the epsilon network
        self.scheduler = scheduler # Take in the noise scheduler

    def _scheduled_call(self, x_0):

        """
        Randomly samples some noised data given some initial x_0, with given scheduler
        """

        t = torch.randint(low=0, high=self.scheduler.steps +1, size=(x_0.size(0),1), device=x_0.device) # +1 at max to get T as well, get T as (Batch,1)
        noise, x_T = self.scheduler.add_noise(x_0, t)

        # Return both the random time and the noised data related to it
        return t, noise, x_T

    def _recover_signal(self, x_T, t, ab):

        """
        Using the epsilon network, abundance condition, and the random noise, recover the actual signal x_0.

        Here, we simply rearrange the noise forward process to have:

        x_0 = (1/sqrt(alpha_t)) * (x_t - sqrt(1 - alpha_t) * noise, or epsilon) 
        
        So, we simply use epsilon to predict how much noise was there on the signal that we got. And the rest is just algebra...
        """

        eps_pred = self.epsilon(x_T, t.to(torch.float32), ab) # Must convert the time to float
        alpha_bar_t = self.scheduler.gather(self.scheduler.alpha_bars, t, x_T.ndim).to(x_T.device)
        x0_pred = (x_T - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        return x0_pred, eps_pred
    
    def training_procedure(self, x_0, ab):

        """
        Given some batched x_0 and ab (abundances related), it returns the predicted x_0. 
        Such predicted x_0 come from an entire procedure of forward noising and denoising using epsilon.
        The returned values will be used to get a loss to then backprop on epsilon (noise prediction) network.
        """

        t, noise, x_T = self._scheduled_call(x_0)
        x0_pred, eps_pred = self._recover_signal(x_T, t, ab)

        return x0_pred, noise, eps_pred

    def sample(self, x_T, ab):

        """
        Sample a signal using the diffusion model.

        Parameters:
            - ab (Tensor); Abundance condition, shape [B, n_ab]
            - x_T (Tensor, optional); Starting noisy signal. If none, Gaussian noise is used.

        Returns:
            - x_0: (Tensor); Generated spectra, shape [B, n_bands]
            - x_T: (Tensor); The high temperature spectrum before denoising, shape [B, n_bands]
        """

        device = ab.device # Get the device using ab 
        B = ab.size(0) # Get the batch number using ab

        steps = self.scheduler.steps # Get the total amounts of steps

        n_bands = self.epsilon.n_bands # Get the bands, used to randomly generate noise

        if x_T is None: # If no x_T is given, define some using gaussian
            x_T = torch.randn(B, n_bands, device=device)
        
        # Reverse diffusion loop, for more detail on DDPM, check the paper on it.

        x_t = x_T # With such redefinition, we make sure that we preserve high temperature spectrum.

        t_s = torch.arange(1, steps + 1) # Time steps from 1 to T
        t_s_reverse = t_s.flip(0) # Get the time steps in reverse order, T to 1

        self.alphas = self.scheduler.alpha_t(torch.cat((torch.tensor([0]), t_s))) # Precompute all the alphas for each time step in order add t=0
        self.betas = self.scheduler.beta_t(torch.cat((torch.tensor([0]), t_s))) # Precompute all the betas for each time step in order add t=0
        self.beta_tildas = self.scheduler.beta_tilda_t(t_s) # Precompute all the beta tildas for each time step in order

        for t in t_s_reverse: # Reversing the range so that: t = T, T-1, T-2, ..., 2, 1
            t_tensor = torch.full((B,1), t, dtype=torch.long, device=device)

            # Predict the noise that was added last step
            eps = self.epsilon(x_t, t_tensor.to(torch.float32), ab)

            # Compute variance (sigma^2) / noise for stochastic sampling
            # For simplicity, use sqrt(beta_tilda_t) for this. The DDPM paper: https://arxiv.org/pdf/2006.11239
            sigma = torch.sqrt(self.scheduler.gather(self.beta_tildas, t_tensor-1, x_t.ndim)).to(torch.float32) # -1 on the time tensor because the 0th index of beta is actually beta tilda at t=1

            # Sample noise, without any noise at step t=0
            z = torch.randn_like(x_t) if t > 0 else 0.0

            # Gather alpha_t
            alpha_t = self.scheduler.gather(self.alphas, t_tensor, x_t.ndim).to(torch.float32)

            # Gather alpha_bar_t
            alpha_bar_t = self.scheduler.gather(self.scheduler.alpha_bars, t_tensor, x_t.ndim).to(torch.float32)

            # Update x_t for the next iter
            x_t = 1/np.sqrt(alpha_t) * (x_t - (1-alpha_t/np.sqrt(1-alpha_bar_t) * eps)) + sigma * z  

        return x_t, x_T
