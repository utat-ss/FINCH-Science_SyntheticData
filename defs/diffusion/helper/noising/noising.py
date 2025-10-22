"""
This file is used to define all the noise scheduling classes.
These are incredibly useful in the training of diffusion
models.
"""

import numpy as np
import torch

from abc import ABC, abstractmethod

# This one has full implementation
class CosSchedule_Old:

    """
    Creates a cosine schedule for the noise.
    Uses the definition in the paper; Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """

    def __init__(self, steps, offset= 8e-3, exp= 2):
        self.steps = steps
        self.offset = offset
        self.exp = exp
        # Precompute all the alpha bars
        self._precompute_alpha_bars()

    def _precompute_alpha_bars(self):

        T = self.steps # Total time, defined by steps
        s = self.offset

        # Definition from the paper, page 4
        times = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((((times / T) + s) / (1.0 + s) ) * torch.pi / 2) ** self.exp # Generate all the f(t) vals, by the def
        f_0 = f[0] # Get the f_0
        self.alpha_bars = torch.tensor(f/f_0, dtype=torch.float32) # Normalize by f_0 and store as alpha bars

    def gather(self, values, t, xndim):
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
        a_bar = self.gather(self.alpha_bars, t, x_0.ndim).to(x_0.device) # Sample the alpha bar at time step t for each item in a batch
        x_T = torch.sqrt(a_bar)*x_0 + torch.sqrt(1.0 - a_bar)*noise # Definition from the paper, page 2

        return noise, x_T
    
    def mu_tilda_t(self, x_0, t):

        """
        Don't exactly know what this is for honestly. Still implemented it.
        """

        x_t = self.add_noise(x_0, t)
        b_t = self.gather(self.beta_t(t), t, x_0.shape).to(x_0.device)
        a_t = 1.0 - b_t

        # Definition from the paper, page 2
        return (torch.sqrt(self.alpha_bars[t-1]) * b_t) / (1.0 - self.alpha_bars[t]) * x_0 + (torch.sqrt(a_t) * (1.0 - self.alpha_bars[t-1])) / (1.0 - self.alpha_bars[t]) * x_t
    

class Schedule(ABC):

    """
    Abstract class for all schedules.
    All schedules should precompute alpha bars in the _precompute_alpha_bars method.
    """

    def __init__(self, steps):
        self.steps = steps
        # Precompute all the alpha bars
        self._precompute_alpha_bars()

    @abstractmethod
    def _precompute_alpha_bars(self):
        pass
    
    """
    The rest of the methods are common to all schedules. They follow the paper: Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """

    def gather(self, values, t, xndim):
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
        a_bar = self.gather(self.alpha_bars, t, x_0.ndim).to(x_0.device) # Sample the alpha bar at time step t for each item in a batch
        x_T = torch.sqrt(a_bar)*x_0 + torch.sqrt(1.0 - a_bar)*noise # Definition from the paper, page 2

        return noise, x_T
    
    def mu_tilda_t(self, x_0, t):

        """
        Don't exactly know what this is for honestly. Still implemented it.
        """

        x_t = self.add_noise(x_0, t)
        b_t = self.gather(self.beta_t(t), t, x_0.shape).to(x_0.device)
        a_t = 1.0 - b_t

        # Definition from the paper, page 2
        return (torch.sqrt(self.alpha_bars[t-1]) * b_t) / (1.0 - self.alpha_bars[t]) * x_0 + (torch.sqrt(a_t) * (1.0 - self.alpha_bars[t-1])) / (1.0 - self.alpha_bars[t]) * x_t

class CosSchedule(Schedule):

    """
    Creates a cosine schedule for the noise.
    Uses the definition in the paper; Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """

    # Needs its own init since it takes extra params
    def __init__(self, steps, offset= 8e-3, exp= 2):
  
        # Take in the non-common params
        self.offset = offset
        self.exp = exp

        # Calls the super init to take in steps and precompute alpha bars
        super().__init__(steps)

    def _precompute_alpha_bars(self):

        T = self.steps # Total time, defined by steps
        s = self.offset

        # Definition from the paper, page 4
        times = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((((times / T) + s) / (1.0 + s) ) * torch.pi / 2) ** self.exp # Generate all the f(t) vals, by the def
        f_0 = f[0] # Get the f_0
        self.alpha_bars = torch.tensor(f/f_0, dtype=torch.float32) # Normalize by f_0 and store as alpha bars

class SqrtSchedule(Schedule):

    """
    Creates a sqrt schedule for the noise.
    Uses the definition in the paper; Li, et al., 2022: https://arxiv.org/pdf/2205.14217, Appendix A
    """

    # No def init needed, doesn't take extra params

    def _precompute_alpha_bars(self):

        T = self.steps # Total time, defined by steps

        # Get the times array
        times = torch.arange(T + 1, dtype=torch.float64)

        # Definition from the paper, appendix A
        self.alpha_bars = 1.0 - torch.sqrt(times / (T))

    """
    The above def is good enough, once it is made sure that the current CosSchedule works, a superclass will be made such that _precompute_alpha_bars is an abstract method.
    """

class LinearSchedule(Schedule):

    """Creates a linear schedule for α's"""
    def __init__(self, steps, alpha_start, alpha_end):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        # Calls the super init to take in steps and precompute alpha bars
        super().__init__(steps)
    
    def _precompute_alpha_bars(self):
        # Precompute the array of alphas, signal retentions, with linear reduction
        self.alphas = torch.linspace(self.alpha_start, self.alpha_end, self.steps +1, dtype=torch.float64)

        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # Must override these since Linear Scheduling is unstable to retreive alpha, beta, beta_tilda from alpha_bar
    def alpha_t(self, t):
        return self.alphas[t]
    
    def beta_t(self, t):
        return 1.0 - self.alphas[t]
    
    def beta_tilda_t(self, t):

        t_safe = torch.clamp(t, min=1)
        
        # log(β̃_t) = log(β_t) + log(1 - ᾱ_{t-1}) - log(1 - ᾱ_t)
        # We use torch.log1p(-x) for log(1-x) to maintain precision when x is close to 1.
        
        log_beta = torch.log(self.beta_t(t_safe))
        log_term1 = torch.log1p(-self.alpha_bars[t_safe - 1])
        log_term2 = torch.log1p(-self.alpha_bars[t_safe])
        
        log_beta_tilda = log_beta + log_term1 - log_term2
        
        return torch.exp(log_beta_tilda)

class ConstantSchedule(Schedule):

    """Creates a constant schedule for α's"""
    def __init__(self, steps, alpha_const):
        self.alpha_const = alpha_const
        # Calls the super init to take in steps and precompute alpha bars
        super().__init__(steps)
    
    def _precompute_alpha_bars(self):
        # Precompute the array of alphas, signal retentions, with constant reduction

        T = self.steps

        self.alphas = self.alpha_const* torch.ones(T + 1, dtype=torch.float64) # Just a constant def

        self.alpha_bars = torch.cumprod(self.alphas, dim=0, dtype=torch.float64)

    # Get these while we are at it
    def alpha_t(self, t):
        return self.alphas[t]
    
    def beta_t(self, t):
        return 1- self.alphas[t]