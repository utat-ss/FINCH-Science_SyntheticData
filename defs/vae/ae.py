"""
Action Item: Autoencoder (Convolutional)

Ideally: Latent dimension = 3 (for gv, npv, soil)
"""

import torch
import torch.nn as nn


class ConvLayers(nn.Module):
    """
    Convolutional layers for encoder and decoder
    """
    def __init__(self, in_dim, h_dim, out_dim, kernel_size=3, stride=1, padding=1, layers=3, pool_kernel=3, pool_stride=1):
        super().__init__()
        self.in_dim = in_dim
        self.conv_hidden_dim = h_dim
        self.out_dim = out_dim
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        # Adding all of the convolution layers, with ReLU and Maxpooling based on inputs
        conv_layers = []
        conv_layers.extend([
            nn.Conv1d(self.in_dim, self.conv_hidden_dim, 
                      kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        ])
        for _ in range(0, layers):
            conv_layers.extend([
                nn.Conv1d(self.conv_hidden_dim, self.conv_hidden_dim, 
                      kernel_size=self.kernel, stride=self.stride, padding=self.padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_stride)
            ])
        conv_layers.extend([
            nn.Conv1d(self.conv_hidden_dim, self.out_dim, 
                      kernel_size=self.kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        ])

        self.net = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.net(x)



class MLP(nn.Module):
    """
    MLP classes for the encoder and decoder
    """
    def __init__(self, in_dim, h_dim=[128, 128], out_dim=3):
        super().__init__()
        self.spec_dim = in_dim
        self.denoiser_hidden_dim = h_dim
        self.out_dim = out_dim

        denoiser_layers = []
        denoiser_layers.extend([
            nn.Linear(self.spec_dim, self.denoiser_hidden_dim[0]),
            nn.ReLU()
        ])
        for i in range(0, len(self.denoiser_hidden_dim) - 1): # Appending hidden layers of denoiser
            denoiser_layers.extend([
                nn.Linear(self.denoiser_hidden_dim[i], self.denoiser_hidden_dim[i+1]),
                nn.ReLU()
            ])
        denoiser_layers.extend([ 
            nn.Linear(self.denoiser_hidden_dim[-1], self.out_dim),
            nn.Sigmoid()
        ]) # Final output layer

        self.net = nn.Sequential(*denoiser_layers)

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):
    """
    The Autoencoder class, intending to combine both ConvLayers and MLP classes.
    """