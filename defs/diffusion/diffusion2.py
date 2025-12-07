"""
Define the diffusion class here
"""

import torch
import torch.nn as nn
import numpy as np

class cond_diffusion(nn.Module):
    def __init__(self, epsilon: nn.Module, scheduler, cfg=None):
        super().__init__()
        self.epsilon = epsilon
        self.scheduler = scheduler
        self.cfg = cfg or {}
        # EMA (if requested)
        self.ema_decay = self.cfg.get("ema_decay", 0.9999)
        self.ema_params = None
        self.registered_ema = False

    def _scheduled_call(self, x_0):

        """
        Randomly samples some noised data given some initial x_0, with given scheduler
        """
                
        # support weighted timestep sampling (scheduler.sample_timesteps)
        if hasattr(self.scheduler, "sample_timesteps"):
            t = self.scheduler.sample_timesteps(x_0.size(0), device=x_0.device)
        else:
            t = torch.randint(low=1, high=self.scheduler.steps + 1, size=(x_0.size(0),), device=x_0.device)
        noise, x_t = self.scheduler.add_noise(x_0, t)
        return t, noise, x_t

    def _recover_signal(self, x_t, ab, t, unconditional_prob=0.0):

        """
        Using the epsilon network, abundance condition, and the random noise, recover the actual signal x_0.

        Here, we simply rearrange the noise forward process to have:

        x_0 = (1/sqrt(alpha_t)) * (x_t - sqrt(1 - alpha_t) * noise, or epsilon) 
        
        So, we simply use epsilon to predict how much noise was there on the signal that we got. And the rest is just algebra...
        """
                
        # epsilon signature supports (x_t, t, ab, unconditional_prob)
        out = self.epsilon(x_t, t, ab, unconditional_prob=unconditional_prob)
        if isinstance(out, tuple):
            eps_pred, logvar = out
        else:
            eps_pred, logvar = out, None
        a_bar = self.scheduler.gather(self.scheduler.alpha_bars.to(x_t.device), t, x_t.ndim)
        x0_pred = (x_t - torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * eps_pred) / torch.sqrt(torch.clamp(a_bar, min=1e-12))
        return x0_pred, eps_pred, logvar

    def training_procedure(self, x_0, ab, unconditional_prob=0.0):

        """
        Given some batched x_0 and ab (abundances related), it returns the predicted x_0. 
        Such predicted x_0 come from an entire procedure of forward noising and denoising using epsilon.
        The returned values will be used to get a loss to then backprop on epsilon (noise prediction) network.
        """
                
        t, noise, x_t = self._scheduled_call(x_0)
        x0_pred, eps_pred, logvar = self._recover_signal(x_t, ab, t, unconditional_prob=unconditional_prob)
        return x0_pred, noise, eps_pred, logvar, t

    def _init_ema(self):
        if self.ema_params is None:
            self.ema_params = [p.detach().cpu().clone() for p in self.epsilon.parameters()]
            self.registered_ema = True

    def update_ema(self):
        self._init_ema()
        with torch.no_grad():
            for ema_p, p in zip(self.ema_params, self.epsilon.parameters()):
                ema_p.mul_(self.ema_decay)
                ema_p.add_((1.0 - self.ema_decay) * p.detach().cpu())

    def load_ema_to_model(self):
        if self.ema_params is None:
            return
        for p, ema_p in zip(self.epsilon.parameters(), self.ema_params):
            p.data.copy_(ema_p.to(p.device))

    def sample(self, ab, x_T=None, timesteps: list = None, eta: float = 0.0, guidance_scale: float = 1.0):
        """
        DDIM-style sampler with optional classifier-free guidance (guidance_scale).

        Sample a signal using the diffusion model.

        Parameters:
            - ab (Tensor); Abundance condition, shape [B, n_ab]
            - x_T (Tensor, optional); Starting noisy signal. If none, Gaussian noise is used.

        Returns:
            - x_0: (Tensor); Generated spectra, shape [B, n_bands]
            - x_T: (Tensor); The high temperature spectrum before denoising, shape [B, n_bands]
        """
        device = ab.device
        B = ab.size(0)
        T = self.scheduler.steps
        n_bands = self.epsilon.n_bands

        if x_T is None:
            x_t = torch.randn(B, n_bands, device=device)
        else:
            x_t = x_T.to(device)

        # build sequence
        if timesteps is None:
            seq = list(range(T, -1, -1))
        else:
            seq = list(timesteps)
            if seq[0] != T:
                raise ValueError("timesteps must start at scheduler.steps (T)")
            if seq[-1] != 0:
                seq = seq + [0]

        for i in range(len(seq) - 1):
            t = seq[i]
            t_prev = seq[i + 1]
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            t_prev_tensor = torch.full((B,), t_prev, dtype=torch.long, device=device)

            # predict eps: for guidance we need both conditional and unconditional predictions
            # unconditional: set unconditional_prob=1.0 to drop conditioning at forward
            out_uncond = self.epsilon(x_t, t_tensor, ab, unconditional_prob=1.0)
            out_cond = self.epsilon(x_t, t_tensor, ab, unconditional_prob=0.0)
            if isinstance(out_uncond, tuple):
                eps_uncond, _ = out_uncond
            else:
                eps_uncond = out_uncond
            if isinstance(out_cond, tuple):
                eps_cond, _ = out_cond
            else:
                eps_cond = out_cond

            # classifier-free guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            a_bar_t = self.scheduler.gather(self.scheduler.alpha_bars.to(device), t_tensor, x_t.ndim)
            a_bar_prev = self.scheduler.gather(self.scheduler.alpha_bars.to(device), t_prev_tensor, x_t.ndim)

            x0_pred = (x_t - torch.sqrt(torch.clamp(1.0 - a_bar_t, min=0.0)) * eps) / torch.sqrt(torch.clamp(a_bar_t, min=1e-12))

            sigma = self.scheduler.ddim_sigma(eta, t_tensor, t_prev_tensor).to(device)  # (B,1)
            # deterministic direction
            coeff = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma ** 2, min=0.0)).expand_as(x_t)
            dir_xt = coeff * eps

            x_prev = torch.sqrt(torch.clamp(a_bar_prev, min=0.0)).expand_as(x_t) * x0_pred + dir_xt

            if float(eta) > 0.0:
                z = torch.randn_like(x_t)
                x_prev = x_prev + sigma.expand_as(x_t) * z

            x_t = x_prev

        return x_t, x_T