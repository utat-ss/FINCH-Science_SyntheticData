"""
This file is used to define all the noise scheduling classes.
These are incredibly useful in the training of diffusion
models.
"""

import numpy as np
    
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

class CosSchedule:
    """
    Creates a cosine schedule for α's
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
        times = np.arange(0, T + 1, dtype=np.float64)
        f = np.cos((((times / T) + s) / (1.0 + s) ) * np.pi / 2) ** self.exp 
        f_0 = f[0]
        self.alpha_bars = f/f_0

    def beta_t(self, t):
        return 1.0 - (self.alpha_bars[t] / self.alpha_bars[t-1]) # Definition from the paper, page 4
    
    def beta_tilda_t(self, t):
        return self.beta_t(t) * (1.0 - self.alpha_bars[t-1]) / (1.0 - self.alpha_bars[t]) # Definition from the paper, page 2
    
    def add_noise(self, x_0, t):
        noise = np.random.randn(*x_0.shape) # Get some random noise of the same shape
        a_bar = self.alpha_bars[t] # Sample the alpha bar at time step t
        return np.sqrt(a_bar)*x_0 + np.sqrt(1.0 - a_bar)*noise # Definition from the paper, page 2
    
    def mu_tilda_t(self, x_0, t):
        x_t = self.add_noise(x_0=x_0, t=t)
        b_t = self.beta_t(t)
        a_t = 1.0 - b_t

        # Definition from the paper, page 2
        return (np.sqrt(self.alpha_bars[t-1]) * b_t) / (1.0 - self.alpha_bars[t]) * x_0 + (np.sqrt(a_t) * (1.0 - self.alpha_bars[t-1])) / (1.0 - self.alpha_bars[t]) * x_t