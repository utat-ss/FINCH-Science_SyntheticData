"""
This file is used to define all the noise scheduling classes.
These are incredibly useful in the training of diffusion
models.
"""

import numpy as np
import torch

from abc import ABC, abstractmethod

class Schedule(ABC):

    """
    Abstract class for all schedules.
    All schedules should precompute alpha bars in the _precompute_alpha_bars method.
    """

    def __init__(self, steps, device=None, timestep_probs=None):
        self.steps = steps
        self.device = device or torch.device("cpu")
        # timestep_probs: optional array-like length steps+1 of sampling weights (for t=0..T).
        # by default we will sample uniformly over 1..T for training.
        if timestep_probs is not None:
            probs = torch.as_tensor(timestep_probs, dtype=torch.float32)
            if probs.numel() != self.steps + 1:
                raise ValueError("timestep_probs must have length steps+1")
            self.timestep_probs = probs.to(self.device)
            # mask out t=0 for training sampling convenience (we don't train on t=0)
            # but keep full distribution available if desired.
        else:
            self.timestep_probs = None
        self._precompute_alpha_bars()
        # derive alphas / betas when not provided
        if not hasattr(self, "alphas"):
            # alphas[t] = alpha_bar[t] / alpha_bar[t-1] (with alpha_bar[-1]=1)
            a_bar = self.alpha_bars
            a_prev = torch.cat([torch.tensor([1.0], device=self.device), a_bar[:-1]])
            self.alphas = torch.clamp(a_bar / a_prev, min=1e-12)
        if not hasattr(self, "betas"):
            self.betas = torch.clamp(1.0 - self.alphas, min=0.0)

    def sample_timesteps(self, batch_size, device=None):
        """
        Sample timesteps for a batch according to self.timestep_probs if provided,
        otherwise uniform over [1..T] (excluding 0).
        Returns a LongTensor shape (batch_size,)
        """
        device = device or self.device
        if self.timestep_probs is None:
            return torch.randint(low=1, high=self.steps + 1, size=(batch_size,), device=device)
        # sample from probs but avoid sampling t=0 if we want training t in [1..T]
        # create a probs tensor for indices 1..T
        probs = self.timestep_probs.clone().to(device)
        # ensure non-negative and normalize
        probs = torch.clamp(probs, min=0.0)
        # zero out t=0 to avoid selecting it for training choice (common choice)
        probs[0] = 0.0
        if probs.sum() <= 0:
            # fallback to uniform if invalid
            return torch.randint(low=1, high=self.steps + 1, size=(batch_size,), device=device)
        probs = probs / probs.sum()
        # torch.multinomial over indices 0..T
        inds = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        return inds.to(torch.long)

    @abstractmethod
    def _precompute_alpha_bars(self):
        pass

    """
    The rest of the methods are common to all schedules. They follow the paper: Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """

    def gather(self, values, t, xndim):
# Needed to make sampled ts compatible with differences within the same batch, essentially makes batches have different sampled ts within them

        # values: tensor indexed by integer timesteps; t: LongTensor shape (B,)
        if isinstance(t, int):
            out = values[t]
        else:
            out = values[t]  # advanced indexing returns (B, ...) depending on values
            # make broadcastable to x with xndim
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

        t_minus1_safe = torch.clamp(t - 1, min=0)

        return torch.clamp(1.0 - (self.alpha_bars[t] / self.alpha_bars[t_minus1_safe]), max=0.99999)

    def alpha_t(self, t):

        """
        Takes in the time tensor.

        Returns α_t = 1 - β_t, how much of the signal is retained
        """
        return self.alphas[t] if hasattr(self, "alphas") else 1.0 - self.beta_t(t)

    def beta_tilda_t(self, t):

        """
        Takes in the time tensor.

        Returns the modified beta_tilda as defined in the paper, page 2
        """

        t_safe = torch.clamp(t, min=1)
        return self.beta_t(t_safe) * (1.0 - self.alpha_bars[t_safe - 1]) / (1.0 - self.alpha_bars[t_safe]) # Definition from the paper, page 2

    def add_noise(self, x_0, t):

        """
        Given some initial signal x_0, add the predicted noise at time t.
        """
        noise = torch.randn_like(x_0, device=x_0.device) # Get some random noise of the same shape
        a_bar = self.gather(self.alpha_bars, t, x_0.ndim).to(x_0.device) # Sample the alpha bar at time step t for each item in a batch
        x_T = torch.sqrt(a_bar) * x_0 + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise # Definition from the paper, page 2
        
        return noise, x_T

    def ddim_sigma(self, eta: float, t, t_prev):
        """
        Compute DDIM sigma for transition from t -> t_prev:
        sigma = eta * sqrt( ((1 - a_bar_prev)/(1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev) )
        Accepts int or LongTensor t and t_prev; returns tensor shaped (B,1) or scalar.
        """
        a_bars = self.alpha_bars
        # convert ints to tensors if needed
        if isinstance(t, int) and isinstance(t_prev, int):
            a_bar_t = a_bars[t]
            a_bar_prev = a_bars[t_prev]
            denom = max(1e-12, 1.0 - float(a_bar_t))
            ratio = max(0.0, (1.0 - float(a_bar_prev)) / denom)
            frac = max(0.0, 1.0 - (float(a_bar_t) / max(1e-12, float(a_bar_prev))))
            sigma = float(eta) * np.sqrt(max(0.0, ratio * frac))
            return torch.tensor(sigma, device=self.device, dtype=torch.float32)
        # batch tensors
        a_bar_t = self.gather(a_bars, t, 1).squeeze(-1)  # (B,)
        a_bar_prev = self.gather(a_bars, t_prev, 1).squeeze(-1)
        denom = torch.clamp(1.0 - a_bar_t, min=1e-12)
        ratio = torch.clamp((1.0 - a_bar_prev) / denom, min=0.0)
        frac = torch.clamp(1.0 - (a_bar_t / torch.clamp(a_bar_prev, min=1e-12)), min=0.0)
        term = ratio * frac
        sigma = float(eta) * torch.sqrt(torch.clamp(term, min=0.0))
        return sigma.unsqueeze(-1)  # (B,1)

    def make_timesteps_subseq(self, n_steps: int):
        """
        Utility: produce a subsequence of timesteps from T down to 0 with n_steps steps (inclusive).
        """
        if n_steps >= self.steps + 1:
            return list(range(self.steps, -1, -1))
        # linear spacing in indices
        idx = np.linspace(0, self.steps, n_steps, dtype=int)[::-1]
        return idx.tolist()

class CosSchedule(Schedule):

    """
    Creates a cosine schedule for the noise.
    Uses the definition in the paper; Nichol et al., 2021: https://arxiv.org/pdf/2102.09672
    """
    def __init__(self, steps, offset= 8e-3, exp= 2, device=None):

        # Take in the non-common params
        self.offset = offset
        self.exp = exp

        # Calls the super init to take in steps and precompute alpha bars
        super().__init__(steps, device=device)

    def _precompute_alpha_bars(self):

        T = self.steps # Total time, defined by steps
        s = self.offset

        # Definition from the paper, page 4
        times = torch.arange(0, T + 1, dtype=torch.float64)
        f = torch.cos(((times / float(T) + s) / (1.0 + s)) * torch.pi / 2.0) ** self.exp # Generate all the f(t) vals, by the def
        f_0 = f[0] # Get the f_0
        self.alpha_bars = (f / f_0).to(torch.float32) # Normalize by f_0 and store as alpha bars

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