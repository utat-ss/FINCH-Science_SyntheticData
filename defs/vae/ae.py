import torch
import torch.nn as nn


class VEncoder(nn.Module):
    """
    Encoder for VCCAE
    """
    def __init__(self, conv_layers, mlp_layers, num_spectra, out_layer, ab_mlp, conv_details):
        super(VEncoder, self).__init__()
        self.conv_details = conv_details
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        self.output_layer = out_layer
        self.abundance_mlp_layers = ab_mlp

        encoder_layers = []
        # Convolutional layers
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
        # MLP layers
        encoder_layers.extend([
            nn.Flatten(start_dim=0), # Flatten convolution to 1 dimension for MLP
            nn.Linear((self.conv_layers[-1] * num_spectra), self.mlp_layers[0]),
            nn.ReLU(),
        ])
        for i in range(0, len(self.mlp_layers) - 1):
            encoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i+1]),
                nn.ReLU(),
            ])
        self.encoder_0 = nn.Sequential(*encoder_layers)
        self.encoder_var = nn.Linear(self.mlp_layers[-1], self.output_layer) # Encodes variation vector
        self.encoder_1 = nn.Linear(self.mlp_layers[-1], self.output_layer) # Last layer of encoder

    def forward(self, x):
        k = self.encoder_0(x)
        variation = self.encoder_var(k) # Produce some variation with respect to the inputted spectrum
        encoded = self.encoder_1(k)
        return encoded, variation
    

class VDecoder(nn.Module):
    """
    Decoder for VCCAE
    """
    def __init__(self, conv_layers, mlp_layers, num_spectra, out_layer, ab_mlp, conv_details, latent_dim):
        super(VDecoder, self).__init__()
        self.conv_details = conv_details
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        self.output_layer = out_layer
        self.abundance_mlp_layers = ab_mlp
        self.latent_dim = latent_dim

        decoder_layers = []
        # MLP layers
        decoder_layers.extend([ # First layer will have size of latent vector space of encoder, plus space of conditioning
                nn.Linear(self.latent_dim, self.mlp_layers[-1]),
                nn.ReLU()
            ])
        for i in range(len(self.mlp_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i-1]),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.Linear(self.mlp_layers[0], (self.conv_layers[-1] * num_spectra)),
            nn.ReLU(),
            nn.Unflatten(0, unflattened_size=(conv_layers[-1], num_spectra)) # Unflatten layer for convolution
        ])
        # Convolutional layers
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
            nn.ConvTranspose1d(self.conv_layers[1], self.conv_layers[0],
                                kernel_size=self.conv_details['k_size'],
                                stride=self.conv_details['pool_stride'],
                                padding=self.conv_details['pad'],
                                output_padding=self.conv_details['out_pad']),
            # nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class VConditioner(nn.Module):

    def __init__(self, conv_layers, mlp_layers, num_spectra, out_layer, ab_mlp, conv_details):
        super(VConditioner, self).__init__()
        self.conv_details = conv_details
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        self.output_layer = out_layer
        self.abundance_mlp_layers = ab_mlp
        
        # Abundance MLP
        ab_mlp_layers = []
        for i in range(0, len(self.abundance_mlp_layers) - 2):
            ab_mlp_layers.extend([
                nn.Linear(self.abundance_mlp_layers[i], self.abundance_mlp_layers[i+1]),
                nn.ReLU()
            ])
        # Last layer will be something else (sigmoid or otherwise)
        ab_mlp_layers.extend([
            nn.Linear(self.abundance_mlp_layers[-2], self.abundance_mlp_layers[-1]),
            # nn.Sigmoid(),
            nn.Flatten(start_dim = 0)
        ])
        self.abundance_adjust = nn.Sequential(*ab_mlp_layers)

    def forward(self, encoded, abundance):
        """
        Forward method for conditioner.

        Applies an MLP on inputted abundances, and concatenates the vector onto the encoded latent vector to produce
        a conditioned vector.
        """
        return torch.cat((encoded, self.abundance_adjust(abundance))) 


class VCCAE(nn.Module):
    """
    Will use VEncoder, VConditioner and VDecoder classes, and playing around with their layers

    Args:
        - conv_layers: Both size and number of convolution layers
        - mlp_layers: Size and number of layers in MLP
        - num_spectra: number of individual data points to be taken. (Default at 210)
        - out_layer: Size of output layer
        - c_d: Contains information about the convolution's parameters. (Included in definition below)
        - ab_mlp: Size and number of conditioner MLP layers.
    """
    def __init__(self, conv_layers:list[int], mlp_layers:list[int], num_spectra=210, out_layer:int = 3,
                 c_d:dict = {}, ab_mlp = [3, 64, 10], scheduler=None):
        super(VCCAE, self).__init__()
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
        self.abundance_mlp_layers = ab_mlp
        self.latent_dim = self.output_layer + self.abundance_mlp_layers[-1]

        for i in c_d:
            self.conv_details[i] = c_d[i] #Any changes to the details of the convolutional layers made here

        self.encoder = VEncoder(conv_layers=self.conv_layers, mlp_layers=self.mlp_layers, num_spectra=num_spectra,
                                out_layer=self.output_layer, ab_mlp=self.abundance_mlp_layers, 
                                conv_details=self.conv_details)
        self.decoder = VDecoder(conv_layers=self.conv_layers, mlp_layers=self.mlp_layers, num_spectra=num_spectra,
                                out_layer=self.output_layer, ab_mlp=self.abundance_mlp_layers, 
                                conv_details=self.conv_details, latent_dim=self.latent_dim)
        self.conditioner = VConditioner(conv_layers=self.conv_layers, mlp_layers=self.mlp_layers, num_spectra=num_spectra,
                                out_layer=self.output_layer, ab_mlp=self.abundance_mlp_layers, 
                                conv_details=self.conv_details)
        
    def forward(self, spectrum, abundance):
        """
        Forward method for VCCAE.

        Will acquire an encoded latent and variation vectors due to encoder, adding a deviation with respect to the variation.
        It will then condition the varied latent vector with its abundances of gv, npv and soil.
        Then, will decode and return 3 values: The decoded spectrum, the encoded latent vector and its variation vector.
        """
        encoded, variation = self.encoder(spectrum)
        deviation = torch.exp(0.5 * variation) # Exponent applied to half of each element in the variation
        distribution = torch.randn_like(deviation) # Normal distribution with mean 0, variance 1, same size as deviation
        z = encoded + (distribution * deviation) # Variation applied
        adjusted = self.conditioner(z, abundance) # Conditioning applied
        decoded = self.decoder(adjusted) # Decoded for final result
        return decoded, encoded, variation
    

"""
Outdated models
"""
class ConvEncoder(nn.Module):
    """
    Convolutional layers for encoder and decoder 
    NOTE: (Outdated)
    """
    def __init__(self, conv_layers:list[int], mlp_layers:list[int], num_spectra=210, out_layer:int = 3,
                 c_d:dict = {}, ab_mlp = [3, 64, 10]):
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
        self.abundance_mlp_layers = ab_mlp

        for i in c_d:
            self.conv_details[i] = c_d[i] #Any changes to the details of the convolutional layers made here

        # Encoder
        encoder_layers = []
        # Convolutional layers
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
        # MLP layers
        encoder_layers.extend([
            nn.Flatten(start_dim=0), # Flatten convolution to 1 dimension for MLP
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


        # Decoder
        decoder_layers = []
        # MLP layers
        decoder_layers.extend([ # First layer will have size of latent vector space of encoder, plus space of conditioning
                nn.Linear(self.output_layer + self.abundance_mlp_layers[-1], self.mlp_layers[-1]),
                nn.ReLU()
            ])
        for i in range(len(self.mlp_layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(self.mlp_layers[i], self.mlp_layers[i-1]),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.Linear(self.mlp_layers[0], (self.conv_layers[-1] * num_spectra)),
            nn.ReLU(),
            nn.Unflatten(0, unflattened_size=(conv_layers[-1], num_spectra)) # Unflatten layer for convolution
        ])
        # Convolutional layers
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
            nn.ConvTranspose1d(self.conv_layers[1], self.conv_layers[0],
                                kernel_size=self.conv_details['k_size'],
                                stride=self.conv_details['pool_stride'],
                                padding=self.conv_details['pad'],
                                output_padding=self.conv_details['out_pad']),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        ])
        self.decoder = nn.Sequential(*decoder_layers)


        # Abundance MLP
        ab_mlp_layers = []
        for i in range(0, len(self.abundance_mlp_layers) - 2):
            ab_mlp_layers.extend([
                nn.Linear(self.abundance_mlp_layers[i], self.abundance_mlp_layers[i+1]),
                nn.ReLU()
            ])
        # Last layer will be something else (sigmoid or otherwise)
        ab_mlp_layers.extend([
            nn.Linear(self.abundance_mlp_layers[-2], self.abundance_mlp_layers[-1]),
            # nn.Sigmoid(),
            nn.Flatten(start_dim = 0)
        ])
        self.abundance_adjust = nn.Sequential(*ab_mlp_layers)

    def forward(self, x, y):
        encoded = self.encoder(x)
        abundance_adjustment = self.abundance_adjust(y)
        encoded = torch.cat((encoded, abundance_adjustment)) # Conditioning
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def adjust(self, x, y): # Concatenate encoded x and conditioning on y
        return torch.cat((x, self.abundance_adjust(y)))
