"""
Define the diffusion class here
"""
import time

import torch
import torch.nn as nn
from .noise.noise_scheduling import *
from .noise.noise_sampling import *

import copy
from contextlib import contextmanager

class GaussianDiffusion(nn.Module):

    def __init__(
            self, 
            epsilon:(nn.Module), 
            augmenter:(nn.Module), 
            scheduler:(Schedule), 
            t_sampler:(Sampling), 
            compressed_sampling:(bool)=False, 
            dynamicthresh_sampling:(bool)=True, 
            sampling_method:(str)='ddpm',
            temperature:(float)=0.8,
            guidance_scale:(float)=2.0,
            uncond_prob:(float)=0.1,
            ema_decay:(float)= 0.999,
            ddim_steps:(int)=50
        ):
        """
        The Denoising Diffusion Probabilistic Model

        Args:
            epsilon: Network to predict noise, conditioned
            augmenter: Data augmenter
            scheduler: Noise scheduler
            t_sampler: Temperature sampler
            compressed_sampling (bool): If want to use float16 for sampling, this makes sampling faster with lower accuracy
            dynamicthresh_sampling (bool): If we want to apply dynamic thresholding during our smapling process
            sampling_method (str): Uses either 'ddpm' backend or 'ddim' backend as a sampling method
            temperature (float): The temperature of the sampler
            guidance_scale (float): The scale for clasifier free guidance
            uncond_prob (float): Probability that during training conditional input gets dropped
            ema_decay (float): The decay amount for Exponential Moving Average (EMA), keep in mind memory = 1/(1-ema_decay)
            ddim_steps (int): Step amount for ddim sampling option
        """
        super().__init__()

        self.epsilon = epsilon # Take in the epsilon network
        self.augmenter = augmenter # Take in the data augmenter
        self.scheduler = scheduler # Take in the noise scheduler
        self.t_sampler = t_sampler # Take in the temp scheduler
        self.compressed_sampling = compressed_sampling # Boolean for compressing the epsilon of sampling step to float16
        self.dynamicthresh_sampling = dynamicthresh_sampling # Boolean for dynamically thresholding during sampling
        self.sampling_method = sampling_method # The sampling method, either ddpm or ddim
        self.temp = temperature # Temperature to change the chaoticity of the model during training, 1 is full
        self.guidance_scale = guidance_scale # The guidance scale to change the 'importance' of conditions
        self.uncond_prob = uncond_prob # Unconditionality probability during training
        self.ema_decay = ema_decay # The amount of decay for t-1 step of EMA while adding t step of model
        self.ddim_steps = ddim_steps # If DDIM is passed as an option for sampling method, this is used to determine delta_t

        # Precompute all the scheduler params. Here, T=steps
        steps = self.scheduler.steps # Get the total amounts of steps from the scheduler
        t_s = torch.arange(steps + 1); self.register_buffer('t_s', t_s) # Time steps from 1 to T. Gets: [0, 1, 2, 3, ..., T-1, T]
        self.register_buffer('t_s_reverse', t_s.flip(0)) # Get the time steps in reverse order, T to 1. Gets: [T, T-1, ..., 3, 2, 1, 0]
        self.register_buffer('alphas', self.scheduler.alpha_t(t_s)) # Precompute all the alphas for each time step in order. This is [T+1]
        self.register_buffer('alpha_bars', self.scheduler.alpha_bars) # This is [T+1]
        self.register_buffer('betas', self.scheduler.beta_t(t_s)) # Precompute all the betas for each time step in order. This is [T+1]
        self.register_buffer('beta_tildas', self.scheduler.beta_tilda_t(t_s)) # Precompute all the beta tildas for each time step in order. This is [T+1]

        # Coeffs for sampling
        self.register_buffer('sampcoef_x', 1.0 / torch.sqrt(self.alphas)) # 1/sqrt(α). This is [T+1]
        betas_safe = self.betas[1:]; alpha_bars_safe = self.alpha_bars[1:]; sampcoef_eps_x_safe = (betas_safe)/torch.sqrt(1-alpha_bars_safe) 
        self.register_buffer('sampcoef_eps_x', torch.cat((torch.tensor([0.0]), sampcoef_eps_x_safe))) # (1-α)/sqrt(1-ᾱ). This is [T+1]
        self.register_buffer('sampcoef_sigma', torch.sqrt(self.beta_tildas)) # the sigma, stdev. This is [T+1]

        # Coeffs for training
        self.register_buffer('traincoef_eps_x', torch.sqrt(1.0-self.alpha_bars))
        self.register_buffer('traincoef_div', torch.sqrt(self.alpha_bars))

        # Get the bands from the epsilon
        self.n_bands = self.epsilon.n_bands # Get the bands, used to randomly generate noise

        # Get an EMA copy at the start, we'll be using this to track Exponential Moving Average
        self.epsilon_ema = copy.deepcopy(self.epsilon).eval().requires_grad_(False)
        self.ema_step = 0 # Set the step count to track how many times we have updated ema

    def _get_coef_at_t(self, buffer:(torch.Tensor), t:(torch.Tensor), x_ndim:(int)):
        """
        A fast and vectorized way to access values in a buffer given a temperature tensor
        
        Args:
            buffer (torch.Tensor): A buffer of shape [Total Steps]
            t (torch.Tensor): Temperature tensor of shape (B,1) where each entry is different
            x_ndim (int): Number of dims for x
        """
        out = buffer[t] # Just access the values

        if x_ndim > 2: # If we have maybe more than 1 channels with signal of shape (B, Ch, Bands), handle that case
            view_shape = [t.shape[0]] + [1] * (x_ndim - 1)
            out = out.view(*view_shape)
        return out

    def _add_noise(self, x_0:(torch.Tensor)):

        """
        Randomly samples some temperature, adds some noise to it, returns t (B,1), noise (B, n_bands), x_t (B, n_bands)
        """

        # Take in some temperature matrix using the temperature sampler
        t = self.t_sampler(x_0) # Give the sampler the x_0 size, and get a t matrix size of (Batch,1)
        noise = torch.randn_like(x_0, device=x_0.device) # Get some random noise of the same shape

        sqrt_alpha_bar = self._get_coef_at_t(self.traincoef_div, t, x_0.ndim)
        sqrt_minus1_alpha_bar = self._get_coef_at_t(self.traincoef_eps_x, t, x_0.ndim)

        x_t = (sqrt_alpha_bar*x_0 + sqrt_minus1_alpha_bar*noise) # Definition from the paper, page 2: Nichol et al., 2021, https://arxiv.org/pdf/2102.09672

        # Return both the random time and the noised data related to it
        return t, noise, x_t.to(x_0.dtype)

    def _recover_signal(self, x_t, t, ab):

        """
        Using the epsilon network, abundance condition, and the random noise, recover the actual signal x_0.

        Here, we simply rearrange the noise forward process to have:

        x_0 = (1/sqrt(alpha_bar_t)) * (x_t - sqrt(1 - alpha_bar_t) * noise, or epsilon)
        
        So, we simply use epsilon to predict how much noise was there on the signal that we got. And the rest is just algebra...
        """

        eps_pred = self.epsilon(x_t, t.to(x_t.dtype), ab) # Must convert the time to float
        coef_eps_x = self._get_coef_at_t(self.traincoef_eps_x, t, x_t.ndim)
        coef_div =self._get_coef_at_t(self.traincoef_div, t, x_t.ndim)
        x0_pred = (x_t - coef_eps_x * eps_pred) / coef_div
        
        return x0_pred, eps_pred
    
    def training_procedure(self, x_0, ab):
        """
        Given some batched x_0 and ab (abundances related), it returns the predicted x_0. 
        Such predicted x_0 come from an entire procedure of forward noising and denoising using epsilon.
        The returned values will be used to get a loss to then backprop on epsilon (noise prediction) network.
        """
        x_0 = self.augmenter(x_0) # Augment the inputted data
        t, noise, x_t = self._add_noise(x_0) # Add some noise to it randomly

        # Logic for conditional droping probability
        if self.uncond_prob > 0.0 and self.training: # If we have such a probability and in the training mode, drop the abundance
            mask = torch.rand(ab.size(0), 1, device=ab.device) < self.uncond_prob # Allows non uniform dropping within a batch
            ab_masked = torch.where(mask, torch.zeros_like(ab), ab) # If dropped, get zeros like, if not, get the full thing
        else: # If we are not dropping at all, just pass in the undropped abundances
            ab_masked = ab 

        x_0_hat, eps_pred = self._recover_signal(x_t, t, ab_masked) # Recover the signal and pred the noise

        return x_0_hat, x_0, eps_pred, noise, t

    def _sample_step_ddpm(self, x_t, t, ab):
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

        eps = self._cfg_forward(x_t, t_tensor, ab)

        coef_x = (self._get_coef_at_t(self.sampcoef_x, t_tensor, x_t.ndim)).to(x_t.dtype) # 1/sqrt(α)
        coef_eps_x = (self._get_coef_at_t(self.sampcoef_eps_x, t_tensor, x_t.ndim)).to(x_t.dtype) # (1-α)/sqrt(1-ᾱ)
        coef_sigma = (self._get_coef_at_t(self.sampcoef_sigma, t_tensor, x_t.ndim)).to(x_t.dtype) # Sigma, equivalent to sqrt(beta_tilda) as per the DDPM paper: https://arxiv.org/pdf/2006.11239

        if self.dynamicthresh_sampling:
            eps = self._dynamic_threshold(x_t, eps, t_tensor) # Applies the threshold in the "x_0 space"

        mu_t = coef_x * (x_t - coef_eps_x * eps)

        # Sample noise, without any noise at step t=0
        z = torch.randn_like(x_t) if t > 1 else 0.0

        # Update x_t for the next iter, multiply z by temperature
        x_t = mu_t + coef_sigma * (z * self.temp)

        return x_t
    
    def _sample_step_ddim(self, x_t, t_now, t_next, ab):
        """
        Sampling step using the DDIM logic. Takes the x_t sample at t_now/t_1/t_current and samples x_t at 
        t_next/t_2/t_target; such that t_1 > t_2. This is done by taking x_(t_1) to x_0 then back to x_(t_2).

        Args:
            x_t: The sample at t_now/t_1/t_current
            t_now: The current time the sample is at, equivalent to t_1/t_current
            t_next: The next time we want the sample sampled at, equivalent to t_2/t_target
            ab: The abundance condition vector
            
        Returns:
            x_target: The sampled x_t where t = t_2/t_next/t_target
        """
        # Get the tensors of current time (t_1) and the next time (t_2) such that t_1 > t_2
        t_tensor_current = torch.full((x_t.size(0),1), t_now, dtype=torch.long, device=x_t.device)
        t_tensor_target = torch.full((x_t.size(0),1), t_next, dtype=torch.long, device=x_t.device)

        # Predict the noise of the sample at t_current/t_now, apply classifier free guidance if needed
        eps = self._cfg_forward(x_t, t_tensor_current, ab)

        # Apply dynamiic threshing if it exists
        if self.dynamicthresh_sampling:
            eps = self._dynamic_threshold(x_t, eps, t_tensor_current)

        # Get the sqrt(alpha_bar) and sqrt(1-alpha_bar) for both t_current and t_target
        sqrt_alpha_bar_current = self._get_coef_at_t(self.traincoef_div, t_tensor_current, x_t.ndim)
        sqrt_minus_alpha_bar_current = self._get_coef_at_t(self.traincoef_eps_x, t_tensor_current, x_t.ndim)
        sqrt_alpha_bar_target = self._get_coef_at_t(self.traincoef_div, t_tensor_target, x_t.ndim)
        sqrt_minus_alpha_bar_target = self._get_coef_at_t(self.traincoef_eps_x, t_tensor_target, x_t.ndim)
    
        # Predict the x_0 given current time (t_1/t_now/t_current)
        pred_x_0 = (x_t - sqrt_minus_alpha_bar_current * eps) / sqrt_alpha_bar_current

        # Now, assuming that we have the clear x_0, we want to get x_t at t_2/t_next/t_target
        dir_x_t = sqrt_minus_alpha_bar_target * eps # Predict the direction that will change x_0 to x_(t_2)
        x_target = sqrt_alpha_bar_target * pred_x_0 + dir_x_t # Add the change
        
        return x_target # Return the target x_(t_2/t_next/t_target)

    def _cfg_forward(self, x_t, t_tensor, ab):
        """
        A forward process to get the epsilon, for the sampling step. This function allows the option to use CFG.
        It also has methods to conduct compressed sampling i.e. running the epsilon network in float16.
        """

        if self.guidance_scale > 1.0: # If guidance scale is more than one, return somewhat unquided epsilon

            x_in = torch.cat([x_t, x_t])
            t_in = torch.cat([t_tensor, t_tensor]).to(x_t.dtype)
            ab_in = torch.cat([ab, torch.zeros_like(ab)])

            if self.compressed_sampling:
                with torch.autocast(device_type=str(x_t.device), dtype=torch.float16): # Using float16, sample
                    eps = self.epsilon(x_in, t_in, ab_in)
                    eps = eps.to(x_t.dtype)
            else:
                eps = self.epsilon(x_in, t_in.to(x_t.dtype), ab_in)

            if eps.ndim==3: eps=eps.squeeze(1) # Squeeze the channel dim of our eps, if we are getting (B, ch, n_bands) as out from it

            eps_cond, eps_uncond = eps.chunk(2)

            return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        
        elif self.guidance_scale == 1.0: # If guidance scale is 1, return the regular, 1 scale guided sampling

            if self.compressed_sampling:
                with torch.autocast(device_type=str(x_t.device), dtype=torch.float16): # Using float16, sample
                    eps = self.epsilon(x_t, t_tensor.to(x_t.dtype), ab)
                    eps = eps.to(x_t.dtype)
            else:
                eps = self.epsilon(x_t, t_tensor.to(x_t.dtype), ab)

            if eps.ndim==3: eps=eps.squeeze(1) # Squeeze the channel dim of our eps, if we are getting (B, ch, n_bands) as out from it

            return eps

        else:
            raise ValueError(f"Unknown guidance scale: {self.guidance_scale}. Must be >= 1.0")

    def _dynamic_threshold(self, x_t, eps, t_tensor):

        """
        Applies some thresholding in the spectrum space
        """

        # Take in the original dtype of x
        init_dtype = x_t.dtype

        # Cast them to float32 because we'll be doing a lot of very critical math cals
        x_t = x_t.to(torch.float32)
        eps = eps.to(torch.float32)

        # infer sqrt(ᾱ) and sqrt(1-ᾱ) 
        sqrt_alpha_bar = self._get_coef_at_t(self.traincoef_div, t_tensor, x_t.ndim).to(torch.float32)
        sqrt_one_minus_alpha_bar = self._get_coef_at_t(self.traincoef_eps_x, t_tensor, x_t.ndim).to(torch.float32)

        # Predict the actual x0 from the given timestep
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar

        # Assume some percentile for which entries they'll be threshed
        static_thresh = 4.0  
        percentile = 0.995

        # Flatten the whole thing
        pred_x0_flat = pred_x0.abs().reshape(pred_x0.shape[0], -1)

        # Get the vals that are in quantile
        s = torch.quantile(pred_x0_flat, percentile, dim=1)
        s = torch.maximum(s, torch.full_like(s, static_thresh))

        # Reshape it
        s = s.view(-1, *([1]*(pred_x0.ndim-1)))

        # Clamp it hard
        pred_x0 = torch.clamp(pred_x0, -s, s) 
        
        # Rescale everything
        pred_x0 = pred_x0 * (static_thresh / s)

        # Get back the epsilon predictions
        eps_new = (x_t - sqrt_alpha_bar * pred_x0) / sqrt_one_minus_alpha_bar

        return eps_new.to(init_dtype)

    def sample(self, ab, x_T=None):
        """
        Sample a signal using the diffusion model.

        Args:
            ab (torch.Tensor); Abundance condition, shape [B, n_ab]
            x_T (torch.Tensor, optional): Starting noisy signal. If none, Gaussian noise is used.

        Returns:
            x_0 (torch.Tensor): Generated spectra, shape [B, n_bands]
            x_T (torch.Tensor): The high temperature spectrum before denoising, shape [B, n_bands]
        """
        device = ab.device # Get the device using ab
        B = ab.size(0) # Get the batch number using ab

        if x_T is None: # If no x_T is given, define some using gaussian
            x_T = torch.randn(B, self.n_bands, device=device)

        # Reverse diffusion loop, for more detail on DDPM, check the paper on it.
        x_t = x_T # With such redefinition, we make sure that we preserve high temperature spectrum.

        if self.sampling_method == 'ddpm':
            for t in self.t_s_reverse[:-1]: # Reversing the range so that: t = T, T-1, T-2, ..., 2, 1. By slicing, we prevent t=0, which is when we have no noise.
                x_t = self._sample_step_ddpm(x_t, t, ab)
        
        elif self.sampling_method == 'ddim':
            step_size = self.scheduler.steps // self.ddim_steps
            seq = list(range(0, self.scheduler.steps + 1, step_size))
            seq = list(reversed(seq)) # Ensure we end at 0
            # Iterate pairs: (1000, 980), (980, 960)...
            for i in range(len(seq) - 1):
                t_now = seq[i]
                t_next = seq[i+1] # This is t-1 in the formula
                x_t = self._sample_step_ddim(x_t, t_now, t_next, ab)

        return x_t, x_T

    def update_ema(self):
        """
        Update the EMA epsilon we are keeping track of, given the current parameters of epsilon
        """
        self.ema_step += 1
        with torch.no_grad():
            # Iterate over both the online parameters and the EMA parameters
            for p_online, p_ema in zip(self.epsilon.parameters(), self.epsilon_ema.parameters()):
                # Update by simply:
                # p_ema = beta * p_ema + p_online * (1 - beta). Where beta = ema_decay
                p_ema.data.mul_(self.ema_decay).add_(p_online.data, alpha= 1 - self.ema_decay)

    @contextmanager
    def use_ema(self):
        """
        Context manager to handle swapping the EMA parameters in and out, so that we can sample with EMA params if needed.
        """
        # Backup the online version of epsilon to the cpu, to keep vram usage lower
        # We cannot simply do a deep copy since it copies everything from vram to vram
        online_backup = {key: value.cpu() for key, value in self.epsilon.state_dict().items()} # A very large dict of epsilon
        self.epsilon.load_state_dict(self.epsilon_ema.state_dict()) # Load the EMA version of epsilon to the class epsilon
        try:
            yield # Within the context, use the EMA parameters
        finally:
            self.epsilon.load_state_dict(online_backup) # Once we are done with the context, reload the online epsilon version
