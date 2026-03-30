import torch
import torch.nn as nn


#region Code rewritten by Ege:

from diffusers.models.activations import get_activation

class VariationalEncoder(nn.Module):
    """
    Variational Encoder for the Conditional-VAE, uses Conv
    
    Args: 
        n_bands (int): Number of bands that will be passed in
        i_ch (int): Number of input channels, usually 1
        latent_dim (int): Size of latent dim
        conv_layers (list[int]): The list of channel sizes for convolution
        kernel_size (int): Kernel size for the conv layers
        stride (int): Stride for the conv layers
        padding (int): Padding for the conv layers
        pool_k (int): The kernel size for pooling
        pool_stride (int): The stride for pooling
        act_fn (str): The str of which act function is being used, must be compatible with diffusers.models.activations.get_activation

    Logic:
        Takes in a data of the form (B, i_ch, seq_len_input)
        
        1- Passes conv layers, (B, i_ch, seq_len_input) -> (B, conv_layers[-1], seq_len_output)
        2- Flattens (B, conv_layers[-1], seq_len_output) -> (B, conv_layers[-1]*seq_len_output)
        3- Passes mlp layers (B, conv_layers[-1]*seq_len_output) -> (B, mlp_layers[-1])
        4- Gets the encoded vals (B, mlp_layers[-1]) -> (B, latent_dim) and (B, latent_dim)
    """

    def __init__(self, n_bands:(int), i_ch:(int), latent_dim:(int), conv_layers:(list[int]), kernel_size:(int), stride:(int), padding:(int), pool_k:(int), pool_stride:(int), mlp_layers:(list[int]), act_fn:(str)):
        super().__init__()

        # Basic init stuff
        # Strategic architectural
        self.i_ch = i_ch
        self.n_bands = n_bands
        self.latent_dim = latent_dim
        self.conv_layers = conv_layers
        self.mlp_layers = mlp_layers
        # Conv related
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_k
        self.pool_stride = pool_stride
        # Misc
        self.act_fn = get_activation(act_fn)

        
        # Building the conv related layers
        seq_len_moving = self.n_bands # This will allow us to track the sequence lengths
        conv_list = [
            nn.Conv1d(
                in_channels=self.i_ch,
                out_channels=self.conv_layers[0],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            ),
            self.act_fn,
            nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
       ]
        # Apply the changes to seq_len
        seq_len_moving = self._get_conv_output_size(seq_len_moving, self.kernel_size, self.stride, self.padding)
        seq_len_moving = self._get_pool_output_size(seq_len_moving, self.pool_kernel_size, self.pool_stride)

        # Serializes the rest of the layers for the Conv
        for i in range(len(self.conv_layers) - 1):
            # Add the layers
            conv_list.extend([
                nn.Conv1d(
                    in_channels=self.conv_layers[i],
                    out_channels=self.conv_layers[i+1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                ),
                self.act_fn,
                nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
            ])
            # Move the seq_len
            seq_len_moving = self._get_conv_output_size(seq_len_moving, self.kernel_size, self.stride, self.padding)
            seq_len_moving = self._get_pool_output_size(seq_len_moving, self.pool_kernel_size, self.pool_stride)

        # Compound everything
        self.conv = nn.Sequential(*conv_list)


        # Build the MLP related layers
        mlp_list = [
            nn.Flatten(start_dim=1), # Flatten on (B, ch, seq_len), gets (B, ch_seq_len)
            nn.Linear(in_features=self.conv_layers[-1] * seq_len_moving, out_features=self.mlp_layers[0]),
            self.act_fn
        ]

        # Serializes the rest of the layers for the MLP
        for i in range(len(self.mlp_layers)-1):
            mlp_list.extend([
                nn.Linear(in_features=self.mlp_layers[i], out_features=self.mlp_layers[i+1]),
                self.act_fn
            ])

        # Compound everything
        self.mlp = nn.Sequential(*mlp_list)

        # Latent dim related
        self.encoder_mu = nn.Linear(in_features=self.mlp_layers[-1], out_features=self.latent_dim)
        self.encoder_logvar = nn.Linear(in_features=self.mlp_layers[-1], out_features=self.latent_dim)

        # Save the final seq_len, will be used by the Decoder
        self.seq_len_final = seq_len_moving

    def _get_conv_output_size(self, seq_len, kernel, stride, padding):
        """Gets the seq_len after applying some convolution, given seq_len, kernel, stride, padding"""
        return ((seq_len + (2*padding) - kernel) // stride) + 1
    
    def _get_pool_output_size(self, seq_len, kernel, stride, padding=0, dilations=1):
        """Gets the seq_len after applying either MaxPool or AvgPool, given kernel, stride, padding=0, dilations=1"""
        return ((seq_len + (2*padding) - dilations*(kernel-1) - 1) // stride) + 1 

    def forward(self, x:(torch.Tensor)):
        """
        Takes in x, gets the mu and logvar

        Args:
            x (torch.Tensor): Of the form either (B, i_ch, seq_len) or (B, seq_len)

        Returns:
            mu (torch.Tensor): Encoded mu that is (B, latent_dim)
            logvar (torch.Tensor): Encoded logvar that is (B, latent_dim)
        """
        if x.ndim == 2: x=x.unsqueeze(1) # Ensures our data is (B, ch, seq_len) while being agnostic to being inputted (B, seq_len) which becomes (B, 1, seq_len)
        
        x = self.conv(x) # Does (B, i_ch, seq_len) -> (B, conv_layers[-1], seq_len_final)
        x = self.mlp(x) # Does (B, conv_layers[-1], seq_len_final) -> (B, conv_layers[-1]*seq_len_final) -> (B, mlp_layers[-1])
        mu = self.encoder_mu(x) # Does (B, mlp_layers[-1]) -> (B, latent_dim)
        logvar = self.encoder_logvar(x) # Does (B, mlp_layers[-1]) -> (B, latent_dim)

        return mu, logvar

class VariationalDecoder(nn.Module):
    """
    Variational Decoder for the Conditional-VAE, uses Conv

    Args:
        n_bands (int): Number of bands in our fed data
        out_ch (int): Outputted spectra's channels
        embedded_dim (int): Embedded dimension size
        seq_len_final (int): The final seq len that was in the encoder
        conv_layers (list[int]): Inverted conv layers of the encoder
        kernel_size (int): Kernel size used in the encoder
        padding (int): Padding used in the encoder
        output_padding (int): Output padding in Transpose convs, usually = 0 since interpolation covers the seq_len mismatches
        pool_stride (int): Pool stride used in encoder
        mlp_layers (list[int]): Inverted mlp layers of Encoder
        act_fn (str):The str of which act function is being used, must be compatible with diffusers.models.activations.get_activation

    Logic:
        Takes in encoded vector with abundance embedding of (B, encoded_dim)

        1- In MLP does (B, embedded_dim) -> (B, conv_layers[0]*seq_len_final)
        2- Unflattens (B, conv_layers[0]*seq_len_final) -> (B, conv_layers[0], seq_len_final) 
        3- Applies Conv layers to squeeze channels, expanding seq_len (B, conv_layers[0], seq_len_final) -> (B, out_ch, seq_len_intermediary)
        4- Applies basic linear interpolation to match sequence lengths from encoder and decoder (B, out_ch, seq_len_intermediary) -> (B, out_ch, seq_len)
        5- If out_ch=1, (B, out_ch, seq_len) -> (B, seq_len)
        6- Applies Sigmoid activation to ensure [0,1]
    """
    def __init__(self, n_bands:(int), out_ch:(int), embedded_dim:(int), seq_len_final:(int), conv_layers:(list[int]), kernel_size:(int), padding:(int), output_padding:(int), pool_stride:(int), mlp_layers:(list[int]), act_fn:(str)):
        super().__init__()
        # Basic init stuff
        self.n_bands = n_bands
        self.out_ch = out_ch
        self.embedded_dim = embedded_dim
        self.seq_len_final = seq_len_final
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding
        self.pool_stride = pool_stride
        self.mlp_layers = mlp_layers
        self.act_fn = get_activation(act_fn)


        # Build MLP layers
        # Init MLP layer
        mlp_list = [
            nn.Linear(in_features=self.embedded_dim, out_features=self.mlp_layers[0]),
            self.act_fn
        ]

        # Serialize the rest of MLP layers
        for i in range(len(self.mlp_layers)-1):
            mlp_list.extend([
                nn.Linear(in_features=self.mlp_layers[i], out_features=self.mlp_layers[i+1]),
                self.act_fn
            ])

        self.seq_len_final = (self.n_bands // self.seq_len_final) * self.seq_len_final # Adjusts the final seq len to be multiple of initial seq_len_final, to ensure proper upsampling later on


        # Final projection before conv
        mlp_list.extend([
            nn.Linear(in_features=self.mlp_layers[-1], out_features=self.seq_len_final*self.conv_layers[0]),
            self.act_fn,
            nn.Unflatten(1, (self.conv_layers[0], self.seq_len_final)) # Unflatten here to do (B, conv_layers[0]*seq_len_final) -> (B, conv_layers[0], seq_len_final)
        ])
        # Compound everything
        self.mlp = nn.Sequential(*mlp_list)


        # Build convolutional layers
        # Can serialize first
        conv_list = []
        for i in range(len(self.conv_layers)-1):
            conv_list.extend([
                nn.ConvTranspose1d(
                    in_channels=self.conv_layers[i],
                    out_channels=self.conv_layers[i+1],
                    kernel_size=self.kernel_size,
                    stride=self.pool_stride,
                    padding=self.padding,
                    output_padding=self.output_padding
                ),
                self.act_fn
            ])

        # Last conv
        conv_list.append(
            nn.ConvTranspose1d(
                in_channels=self.conv_layers[-1],
                out_channels=self.out_ch,
                kernel_size=self.kernel_size,
                stride=self.pool_stride,
                padding=self.padding,
                output_padding=self.output_padding
            )
        )

        # Compound everything
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x:(torch.Tensor)):
        """
        Forward passes a embedded vector x, to decode the spectra
        """
        x = self.mlp(x)
        decoded = self.conv(x)
        if decoded.shape[-1] != self.n_bands: decoded = torch.nn.functional.interpolate(decoded, size=self.n_bands, mode='linear', align_corners=False) # Ensures input to encoder and output from decoder has same n_bands
        if self.out_ch == 1: decoded = decoded.squeeze(1) # Ensures if we are not working with any channels in our dataset, the channel gets squeezed
        return torch.sigmoid(decoded)

class VariationalConditioner(nn.Module):
    """
    Conditions the encoded vector using the abundances

    Args:
        n_endmembers (int): Number of endmembers being considered, usually 3
        embedding_dim (int): The dimensions of final abundance embedding
        embedding_layers (list[int]): The list of embedding hidden layers
        act_fn (str): Self explanatory, same as encoder and decoder
    
    Logic:
        Takes in (B, n_endmembers)

        1- Creates embeddings (B, n_endmembers) -> (B, embedding_dim)
    """
    def __init__(self, n_endmembers:(int), embedding_dim:(int), embedding_layers:(list[int]), act_fn:(str)='relu'):
        super().__init__()
        self.act_fn = get_activation(act_fn)
        ab_list =[]

        # If we want hidden layers
        if len(embedding_layers) > 0:
            # Init layer
            ab_list.extend([
                nn.Linear(n_endmembers, embedding_layers[0]),
                self.act_fn
            ])
            
            # Serialized hidden Layers
            for i in range(len(embedding_layers) - 1):
                ab_list.extend([
                    nn.Linear(embedding_layers[i], embedding_layers[i+1]),
                    self.act_fn
                ])
                
            # Final Projection
            ab_list.append(
                nn.Linear(embedding_layers[-1], embedding_dim)
            )
        else:
            # Direct mapping if no hidden layers
            ab_list.append(
                nn.Linear(n_endmembers, embedding_dim)
            )

        # Compound everything
        self.embedding_layers = nn.Sequential(*ab_list)

    def forward(self, z:(torch.Tensor), ab:(torch.Tensor)):
        """
        Embeds the latent vector z, after creating an abundance embedding

        Args:
            z (torch.Tensor): Latent vector after reparameterization of size (B, latent_dim)
            ab (torch.Tensor): Abundance tensor of size (B, n_endmembers)
        
        Returns:
            embedded (torch.Tensor): Embeds using the embedding vector and latent vector, which gives (B, latent_dim + embedding_dim)
        """

        embedding = self.embedding_layers(ab)

        return torch.cat((z, embedding), dim=1)
    
class VariationalCCAE(nn.Module):
    """
    Uses VariationalEncoder, VariationalDecoder, and VariationalEncoder layers to create a Varitional Conditional Convolutional AutoEncoder
    
    Args:
        conv_layers (list[int]): The list of Encoder and Decoder's CNNs' channels
        mlp_layers (list[int]): The list of Encoder and Decoder's MLPs' layers
        latent_dim (int): The size of the latent vector, before embedding
        embedding_layers (list[int]): The list of abundance embedding's dims, should not have n_endmembers as the first entry
        n_endmembers (int): Number of endmembers
        n_bands (int): Number of bands
        i_ch (int): Inputted amount of channels, usually 1 for us
        out_ch (int): Outputted amount of channels, usually 1 for us
        kernel_size (int): Kernel size for the conv layers
        stride (int): Stride for the conv layers
        padding (int): Padding for the conv layers
        pool_k (int): The kernel size for pooling
        pool_stride (int): The stride for pooling
        output_padding (int): Output padding in Transpose convs, usually = 0 since interpolation covers the seq_len mismatches
    """
    def __init__(
            self,
            conv_layers:(list[int]),
            mlp_layers:(list[int]),
            latent_dim:(int),
            embedding_dim:(int)=10,
            embedding_layers:(list[int])=[64],
            n_endmembers:(int)=3,
            n_bands:(int)=210,
            i_ch:(int)=1,
            out_ch:(int)=1,
            kernel_size:(int)=3,
            stride:(int)=2,
            padding:(int)=1,
            pool_kernel_size:(int)=1,
            pool_stride:(int)=1,
            output_padding:(int)=0,
            act_fn:(str)='relu',
            scheduler=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim + n_endmembers

        self.encoder = VariationalEncoder(
            n_bands=n_bands,
            i_ch=i_ch,
            latent_dim=latent_dim,
            conv_layers=conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_k=pool_kernel_size,
            pool_stride=pool_stride,
            mlp_layers=mlp_layers,
            act_fn=act_fn
        )

        seq_len_final = self.encoder.seq_len_final # Get the final sequence length
        decoder_conv_layers = conv_layers[::-1] # Reverse the layers
        decoder_mlp_layers = mlp_layers[::-1] 
        self.decoder = VariationalDecoder(
            n_bands=n_bands,
            out_ch=out_ch,
            embedded_dim=embedding_dim+latent_dim,
            seq_len_final=seq_len_final,
            conv_layers=decoder_conv_layers,
            kernel_size=kernel_size,
            padding=padding,
            output_padding=output_padding,
            pool_stride=pool_stride,
            mlp_layers=decoder_mlp_layers,
            act_fn=act_fn
        )

        self.conditioner = VariationalConditioner(
            n_endmembers=n_endmembers,
            embedding_dim=embedding_dim,
            embedding_layers=embedding_layers,
            act_fn=act_fn
        )

    def _reparameterize(self, mu:(torch.Tensor), logvar:(torch.Tensor)):
        """Applies the reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, spectrum, abundance):
        """
        Args:
            spectrum (torch.Tensor): The spectra (B, n_bands) or (B, ch, n_bands)
            abundance (torch.Tensor): The abundances (B, n_endmembers)
            
        Returns:
            decoded (torch.Tensor): Reconstructed spectra
            mu (torch.Tensor): Latent mean
            logvar (torch.Tensor): Latent log-variance
        """
        # Encode
        mu, logvar = self.encoder(spectrum)

        # Sample latent vector z
        z = self._reparameterize(mu, logvar)

        # Condition on/embed with abundances
        z_embedded = self.conditioner(z, abundance)

        # Decode
        decoded = self.decoder(z_embedded)

        return decoded, mu, logvar

#endregion

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
