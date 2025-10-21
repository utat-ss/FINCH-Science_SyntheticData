"""
This file is used to define all the noise scheduling classes.
These are incredibly useful in the training of diffusion
models.
"""

import numpy as np
import torch
    
# This has not been implemented in torch yet, and prob is not compatible with the wrapper
class ConstantSchedule:

    """Creates a constant schedule for α's"""
    def __init__(self, steps, alpha_const):
        self.steps = steps # Redundant
        self.alpha_const = alpha_const

    def alpha_bar_t(self, t):
        """Return cumulative α_t at time step t (0 <= t <= steps)"""
        prod = 1.0 # Base def for the product
        for i in range(t+1): # Takes the cumulative product of all α from 0 to a given t
            prod *= self.alpha_const
        return prod
    
    def beta_t(self, t):
        """May be required in the future, returns β_t = 1 - α_t, how much of the signal is lost"""
        return 1.0 - self.alpha_const
    
    def add_noise(self, x0, t):
        """Returns  noisy x_t at time step t"""
        noise = np.random.randn(*x0.shape) # Get some random noise of the same shape
        a_bar = self.alpha_bar_t(t) # Sample the alpha bar at time step t
        return np.sqrt(a_bar) * x0 + np.sqrt(1.0 - a_bar) * noise # Return the final signal, with added noise at time t

# This has not been implemented in torch yet, and prob is not compatible with the wrapper 
class LinearSchedule:

    """Creates a linear schedule for α's"""
    def __init__(self, steps, alpha_start, alpha_end):
        self.steps = steps # Redundant
        # Precompute the array of alphas, signal retentions, with linear reduction
        self.alphas = np.linspace(alpha_start, alpha_end, steps)

    def alpha_t(self, t):
        return self.alphas[t]
    
    def beta_t(self, t):
        """May be required in the future, returns β_t = 1 - α_t, how much of the signal is lost"""
        return 1.0 - self.alphas[t]
    
    def alpha_bar_t(self, t):
        """Return cumulative α_t at time step t (0 <= t <= steps)"""
        prod = 1.0 # Base def for the product
        for i in range(t+1): # Takes the cumulative product of all α from 0 to a given t
            prod *= self.alphas[t]
        return prod
    
    def add_noise(self, x0, t):
        """Returns  noisy x_t at time step t"""
        noise = np.random.randn(*x0.shape) # Get some random noise of the same shape
        a_bar = self.alpha_bar_t(t) # Sample the alpha bar at time step t
        return np.sqrt(a_bar) * x0 + np.sqrt(1.0 - a_bar) * noise # Return the final signal, with added noise at time t

# This one has full implementation
class CosSchedule:

    """
    Creates a cosine schedule for the noise.
    Uses the definition in the paper; Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """

    def __init__(self, steps, offset= 8e-3, exp= 2):
        self.steps = steps
        self.offset = offset
        self.exp = exp
        # Precompute all the alpha
        self._precompute_alpha_bars()

    def _precompute_alpha_bars(self):

        T = self.steps # Total time, defined by steps
        s = self.offset

        # Definition from the paper, page 4
        times = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((((times / T) + s) / (1.0 + s) ) * torch.pi / 2) ** self.exp # Generate all the f(t) vals, by the def
        f_0 = f[0] # Get the f_0
        self.alpha_bars = torch.tensor(f/f_0, dtype=torch.float32) # Normalize by f_0 and store as alpha bars

    def _gather(self, values, t, xndim):
        # Needed to make sampled ts compatible with differences within the same batch, essentially makes batches have different sampled ts within them
        if t.ndim == 0:
            out = values[t]
        else:
            out = values[t]
            while out.ndim < xndim:
                out = out.unsqueeze(-1)
        return out
    
    def beta_t(self, t):

        """
        Takes in the time tensor.

        Returns β_t = 1 - (α_bar_t / α_bar_(t-1)), how much of the signal is lost
        as defined in the paper, page 4

        This one requires 
            - Clamping to 0.99999 at high T to avoid numerical caused singularities
            - Clamping for t=0 to avoid alpha_bars[-1]
        """

        t_minus1_safe = torch.clamp(t-1, min=0) # Clamp the time tensor, so that it can be safely used even when t=0

        if torch.any(t==0):
            return 0 # By definition beta_0 = 0
        
        beta = 1.0 - (self.alpha_bars[t] / self.alpha_bars[t_minus1_safe]) # Definition from the paper, page 4, use clamped time
        return torch.clamp(beta, max=0.99999) # Clamp to 0.99999 max to avoid singularities at high t

    def alpha_t(self, t):

        """
        Takes in the time tensor.

        Returns α_t = 1 - β_t, how much of the signal is retained
        """

        return 1.0 - self.beta_t(t)
        # Simply alpha_t = 1 - beta_t

    def beta_tilda_t(self, t):

        """
        Takes in the time tensor.

        Returns the modified beta_tilda as defined in the paper, page 2
        """

        return self.beta_t(t) * (1.0 - self.alpha_bars[t-1]) / (1.0 - self.alpha_bars[t]) # Definition from the paper, page 2
    
    def add_noise(self, x_0, t):

        """
        Given some initial signal x_0, add the predicted noise at time t.
        """

        noise = torch.randn_like(x_0, device=x_0.device) # Get some random noise of the same shape
        a_bar = self._gather(self.alpha_bars, t, x_0.ndim).to(x_0.device) # Sample the alpha bar at time step t for each item in a batch
        x_T = torch.sqrt(a_bar)*x_0 + torch.sqrt(1.0 - a_bar)*noise # Definition from the paper, page 2

        return noise, x_T
    
    def mu_tilda_t(self, x_0, t):

        """
        Don't exactly know what this is for honestly. Still implemented it.
        """

        x_t = self.add_noise(x_0, t)
        b_t = self._gather(self.beta_t(t), t, x_0.shape).to(x_0.device)
        a_t = 1.0 - b_t

        # Definition from the paper, page 2
        return (torch.sqrt(self.alpha_bars[t-1]) * b_t) / (1.0 - self.alpha_bars[t]) * x_0 + (torch.sqrt(a_t) * (1.0 - self.alpha_bars[t-1])) / (1.0 - self.alpha_bars[t]) * x_t
    

class SqrtSchedule:

    def __init__(self):
        pass