"""
Action Item: Autoencoder (Convolutional)

Ideally: Latent dimension = 3 (for gv, npv, soil)
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional layers for encoder and decoder
    """
    def __init__(self, conv_layers:list[int], mlp_layers:list[int], num_spectra=210, out_layer:int = 3,
                 c_d:dict = {}):
        super().__init__()
        self.conv_details = {
            'k_size': 3,        #kernel size
            'stride': 1,        #
            'pad': 1,           #padding
            'pool_k': 1,        #pool kernel size
            'pool_stride': 1,   #
            'out_pad': 0       #output padding for ConvTranspose1d
        } #Default settings for convolutional layers
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        self.output_layer = out_layer

        for i in c_d:
            self.conv_details[i] = c_d[i] #Any changes to the details of the convolutional layers made here

        # Encoder
        encoder_layers = []
        for i in range(0, len(self.conv_layers) - 1):
            encoder_layers.extend([
                nn.Conv1d(in_channels=self.conv_layers[i], 
                          out_channels=self.conv_layers[i+1], 
                      kernel_size=self.conv_details['k_size'], 
                      stride=self.conv_details['stride'], 
                      padding=self.conv_details['pad']),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.conv_details['pool_k'], stride=self.conv_details['pool_stride']),
            ])
        
        encoder_layers.extend([
            nn.Flatten(start_dim=0),
            nn.Linear((self.conv_layers[-1] * num_spectra), self.mlp_layers[0]),
            nn.ReLU(),
        ])
        for i in range(0, len(self.mlp_layers) - 1):
            encoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i+1]),
                nn.ReLU(),
            ])
        encoder_layers.extend([
                nn.Linear(self.mlp_layers[-1], self.output_layer),
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)


        #Decoder
        decoder_layers = []
        decoder_layers.extend([
                nn.Linear(self.output_layer, self.mlp_layers[-1]),
                nn.ReLU()
            ])
        for i in range(len(self.mlp_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i-1]),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.Linear(self.mlp_layers[0], self.conv_layers[-1]),
            nn.ReLU(),
            nn.Unflatten(0, unflattened_size=(conv_layers[-1], 1))
        ])
        for i in range(len(self.conv_layers) - 1, 1, -1):
            decoder_layers.extend([
                nn.ConvTranspose1d(self.conv_layers[i], self.conv_layers[i-1],
                                   kernel_size=self.conv_details['k_size'],
                                   stride=self.conv_details['pool_stride'],
                                   padding=self.conv_details['pad'],
                                   output_padding=self.conv_details['out_pad']),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.ConvTranspose1d(self.conv_layers[1], (self.conv_layers[0] * num_spectra),
                                kernel_size=self.conv_details['k_size'],
                                stride=self.conv_details['pool_stride'],
                                padding=self.conv_details['pad'],
                                output_padding=self.conv_details['out_pad']),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


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
    conv_details: dict

    def __init__(self, conv_layers:list[int], mlp_layers:list[int], out_layer:int = 3,
                 c_d:dict = {}):
        super().__init__()
        self.conv_details = {
            'k_size': 3,        #kernel size
            'stride': 1,        #
            'pad': 1,           #padding
            'pool_k': 3,        #pool kernel size
            'pool_stride': 1    #
        } #Default settings for convolutional layers
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        self.output_layer = out_layer

        for i in c_d:
            self.conv_details[i] = c_d[i] #Any changes to the details of the convolutional layers made here


        # Encoder
        encoder_layers = []
        for i in range(0, len(self.conv_layers) - 1):
            encoder_layers.extend([
                nn.Conv1d(in_channels=self.conv_layers[i], 
                          out_channels=self.conv_layers[i+1], 
                      kernel_size=self.conv_details['k_size'], 
                      stride=self.conv_details['stride'], 
                      padding=self.conv_details['pad']),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.conv_details['pool_k'], stride=self.conv_details['pool_stride'])
            ])
        # Next, add the MLP using the output layer of the convolution as the input
        encoder_layers.extend([
            nn.Linear(self.conv_layers[-1], self.mlp_layers[0]),
            nn.ReLU()
        ])
        for i in range(0, len(self.mlp_layers) - 1):
            encoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i+1]),
                nn.ReLU()
            ])
        encoder_layers.extend([
                nn.Linear(self.mlp_layers[-1], self.output_layer),
                #nn.Sigmoid()
                # Not needed for encoding?
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)


        #Decoder
        decoder_layers = []
        decoder_layers.extend([
                nn.Linear(self.output_layer, self.mlp_layers[-1]),
                nn.ReLU()
            ])
        for i in range(len(self.mlp_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i-1]),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.Linear(self.mlp_layers[0], self.conv_layers[-1]),
            nn.ReLU()
        ])
        for i in range(len(self.conv_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose1d(self.conv_layers[i], self.conv_layers[i-1],
                                   kernel_size=self.conv_details['k_size'],
                                   stride=self.conv_details['stride'],
                                   padding=self.conv_details['pad'],
                                   output_padding=self.conv_details['pad']),
                nn.ReLU(),
            ])
        decoder_layers.extend([
            nn.ConvTranspose1d(self.conv_layers[1], self.conv_layers[0],
                                kernel_size=self.conv_details['k_size'],
                                stride=self.conv_details['stride'],
                                padding=self.conv_details['pad'],
                                output_padding=self.conv_details['pad']),
            nn.Sigmoid(),
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x) # x represented in the latent space
        return self.decoder(encoded)