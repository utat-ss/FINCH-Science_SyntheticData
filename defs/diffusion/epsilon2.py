"""
Define different epsilons over here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(timesteps, dim):
    # timesteps: (B,) long or float
    half = dim // 2
    device = timesteps.device
    emb = torch.log(torch.tensor(10000.0)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)

class Epsilon_MLP(nn.Module):

    """
    Just a very simple very deep MLP to do noise removal given

    The cfg_model:
    - ['time_embed']['hidden_dim']
    - ['time_embed']['hidden_n']
    - ['ab_embed']['hidden_dim']
    - ['ab_embed']['hidden_n']
    - ['ab_embed']['ab_dim']
    - ['denoiser']['spec_dim'] 
    - ['denoiser']['hidden_dim']
    - ['denoiser']['hidden_n']

    For Forward prop
        - an input spectrum of size [n,]
        - an abundance vector of size [3,]
        - and a time step of size [1,]

    Returns
        - spectrum at time step t-1 of size [n,]
    """
        
    def __init__(self, cfg_model, predict_logvar: bool = False, cond_dropout: float = 0.0):
        super().__init__()

        
        """
        Initializes the network. Takes in the model config.
        """
        
        self.n_bands = cfg_model['denoiser']['spec_dim']
        self.predict_logvar = predict_logvar
        self.cond_dropout = cond_dropout

        # time embedding dimension
        self.time_dim = max(32, cfg_model['time_embed']['hidden_dim'])
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # small projection for time sinusoidal embedding
        self.time_proj_in = nn.Linear(self.time_dim, self.time_dim)

        # abundance embedder
        self.ab_dim = cfg_model['ab_embed']['ab_dim']
        self.ab_hidden = cfg_model['ab_embed']['hidden_dim']
        self.ab_mlp = nn.Sequential(
            nn.Linear(self.ab_dim, self.ab_hidden),
            nn.SiLU(),
            nn.Linear(self.ab_hidden, self.ab_hidden),
            nn.SiLU(),
            nn.Linear(self.ab_hidden, self.ab_hidden),
        )

        # denoiser core
        core_in = self.n_bands + self.time_dim + self.ab_hidden
        self.core_hidden = cfg_model['denoiser']['hidden_dim']
        layers = [nn.Linear(core_in, self.core_hidden), nn.SiLU()]
        for _ in range(cfg_model['denoiser']['hidden_n']):
            layers.append(ResidualBlock(self.core_hidden))
        self.core = nn.Sequential(*layers)

        # outputs
        self.out_eps = nn.Linear(self.core_hidden, self.n_bands)
        self.predict_logvar = predict_logvar
        if self.predict_logvar:
            # two learned extreme log-variance vectors (common stable parameterization)
            # initialize small and large extremes (paper suggests sensible range; tweak if needed)
            self.logvar_small = nn.Parameter(torch.full((self.n_bands,), -10.0))  # near-zero variance
            self.logvar_large = nn.Parameter(torch.full((self.n_bands,), 0.0))    # larger variance
            # project time embedding to a blending coefficient (0..1)
            self.time_to_alpha = nn.Sequential(
                nn.Linear(self.time_dim, self.time_dim),
                nn.SiLU(),
                nn.Linear(self.time_dim, 1),
                nn.Sigmoid()
            )
            # optional per-sample small head if you prefer full predicted logvar:
            # self.out_logvar = nn.Linear(self.core_hidden, self.n_bands)

    def forward(self, x_t, t, ab=None, unconditional_prob: float = 0.0):
        """
        x_t: (B, n_bands)
        t: (B,) long or scalar
        ab: (B, ab_dim) or None
        unconditional_prob: float in [0,1] - for classifier-free guidance training: probability of dropping conditioning
        returns: eps_pred (B, n_bands) and optionally logvar (B, n_bands)
        """
        B = x_t.shape[0]
        if x_t.device != torch.device('cpu') and hasattr(self, "time_proj_in"):
            pass

        # time embedding
        if t.dtype in (torch.int64, torch.long):
            t_float = t.float()
        else:
            t_float = t
        t_emb = sinusoidal_embedding(t_float, self.time_dim).to(x_t.device)
        t_emb = self.time_proj_in(t_emb)
        t_emb = self.time_mlp(t_emb)

        # conditional dropout (classifier-free guidance): randomly drop ab by replacing with zeros
        if ab is None:
            ab_in = torch.zeros((B, self.ab_dim), device=x_t.device)
        else:
            mask = (torch.rand(B, device=x_t.device) < unconditional_prob).float().unsqueeze(-1)
            ab_in = ab * (1.0 - mask)  # dropped entries become 0

        ab_emb = self.ab_mlp(ab_in)

        h = torch.cat([x_t, t_emb, ab_emb], dim=-1)
        h = self.core(h)
        eps = self.out_eps(h)

        if self.predict_logvar:
            # compute interpolation alpha from the time embedding (t_emb)
            # note: we recompute t_emb as in forward flow, you can reuse if refactored
            if t.dtype in (torch.int64, torch.long):
                t_float = t.float()
            else:
                t_float = t
            t_emb = sinusoidal_embedding(t_float, self.time_dim).to(x_t.device)
            t_emb = self.time_proj_in(t_emb)
            t_emb = self.time_mlp(t_emb)
            alpha = self.time_to_alpha(t_emb)  # (B,1) in (0,1)
            # combine extremes
            logvar = alpha * self.logvar_large.unsqueeze(0) + (1.0 - alpha) * self.logvar_small.unsqueeze(0)
            # logvar now shape (B, n_bands)
            return eps, logvar
        return eps