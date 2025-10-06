"""
Define different epsilons over here
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class Epsilon_MLP(nn.Module):

    """
    Just a very simple very deep MLP to do noise removal given

    The cfg_model:
    - ['time_embed']['hidden_dim']
    - ['time_embed']['hidden_n']
    - ['ab_embed']['hidden_dim']
    - ['ab_embed']['hidden_dim']
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

    def __init__(self, cfg_model):
        super().__init__()

        """
        Initializes the network. Takes in the model config.
        """

        self.n_bands = cfg_model['denoiser']['spec_dim'] # This is used in the diffusion class, useful. Represents how many spectral bands there are.

        # Time embedder MLP
        self.time_embed_hidden_dim = cfg_model['time_embed']['hidden_dim'] # Take in the necessary params for the embedder
        self.time_embed_hidden_n = cfg_model['time_embed']['hidden_n']

        time_embed_layers = [] # Define an initial list to be unpacked later on

        time_embed_layers.append(*[ 
            nn.Linear(1, self.time_embed_hidden_dim),
            nn.SiLU(),
        ]) # Appending the init layer

        for _ in range(self.time_embed_hidden_n): # Appending hidden layers
            time_embed_layers.append(*[
                nn.Linear(self.time_embed_hidden_dim, self.time_embed_hidden_dim),
                nn.SiLU()
                ])

        time_embed_layers.append(nn.Linear(self.time_embed_hidden_dim, self.time_embed_hidden_dim)) # Output layer, no actv for unbounded output

        self.time_embed = nn.Sequential(*time_embed_layers) # Compile everything again, by unpacking layer list


        # Abundance embedder MLP
        self.ab_embed_hidden_dim = cfg_model['ab_embed']['hidden_dim'] # Take in the necessary params for the embedder
        self.ab_embed_hidden_n = cfg_model['ab_embed']['hidden_dim']
        self.ab_embed_ab_dim = cfg_model['ab_embed']['ab_dim']

        ab_embed_layers = []

        ab_embed_layers.append(*[
            nn.Linear(self.ab_embed_ab_dim, self.ab_embed_hidden_dim),
            nn.SiLU()
        ]) # Appending the init layer

        for _ in range(self.ab_embed_hidden_n): # Appending hidden layers
            ab_embed_layers.append(*[
                nn.Linear(self.ab_embed_hidden_dim, self.ab_embed_hidden_dim),
                nn.SiLU()
            ]) 

        ab_embed_layers.append(nn.Linear(self.ab_embed_hidden_dim, self.ab_embed_hidden_dim))

        self.ab_embed = nn.Sequential(*ab_embed_layers)


        # Build the full denoiser, that takes in the spectrum itself and the outs from embedders
        self.spec_dim = cfg_model['denoiser']['spec_dim'] 
        self.denoiser_hidden_dim = cfg_model['denoiser']['hidden_dim']
        self.denoiser_hidden_n = cfg_model['denoiser']['hidden_n']

        denoiser_layers = []

        denoiser_layers.append(*[
            nn.Linear(self.spec_dim + self.time_embed_hidden_dim + self.ab_embed_hidden_dim, self.denoiser_hidden_dim),
            nn.SiLU()
        ]) # Init layer of the denoiser, takes in embeddings

        for _ in range(self.denoiser_hidden_n): # Appending hidden layers of denoiser
            denoiser_layers.append(*[
                nn.Linear(self.denoiser_hidden_dim, self.denoiser_hidden_dim),
                nn.SiLU()
            ])

        denoiser_layers.append(nn.Linear(self.denoiser_hidden_dim, self.denoiser_hidden_dim))

        self.denoiser = nn.Sequential(*denoiser_layers)

    def forward(self, x_T, t, ab):

        """
        Inputs:
            - x_T: (B, n_bands)
            - t: (B,) integer time steps
            - ab: (B, n_abund=3) 
        """

        # Unsqueeze to get t to be (B,1)
        t = t.unsqueeze(-1).float()

        # Embed all the conditions
        t_embedded = self.time_embed(t)
        ab_embedded = self.ab_embed(ab)

        # Concat all the conditions with the 
        h = torch.cat([x_T, t_embedded, ab_embedded], dim = -1)

        return self.denoiser(h)
    
