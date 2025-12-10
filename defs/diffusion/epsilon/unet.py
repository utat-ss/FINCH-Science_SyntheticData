import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.activations import get_activation

from .auxiliary.epsilon_addons import Transformer1D, Conformer1D

import math

from typing import Optional, Union
from einops import repeat, pack

"""
The grand majority of this code was taken from: @inproceedings{mehta2024matcha,title={Matcha-{TTS}: A fast {TTS} 
architecture with conditional flow matching}, author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, 
{\'E}va and Henter, Gustav Eje}, booktitle={Proc. ICASSP},year={2024}}

Primary and the most important changes are:
1- Removal of the mask option, since we assume a constant spectral band amount (~210)
2- Removal of the mu conditioning input, and its change to abundances
3- Addition of comments, improved readability. I have added so many comments omg lol
4- Simplification and removal of boilerplate code
"""

class SinusoidalPosEmb(nn.Module):
    """
    Performs sinusoidal embedding, which essentially takes a timestep and expands them into a 
    dimension all the while creating a unique embedding for each timestep.
    Performs time->embedding: (B, ..., 1) -> (B, dim)

    Args:
        dim (int): The output embedding dim, defaulted at 128, usually don't change this.
        scale (int): Total scaling, keep 1 if using a large T range like [0,1k] or [0,5k]
    """
    def __init__(self, dim:(int)=128, scale:(int)=1):
        super().__init__()
        self.dim = dim
        self.scale = scale
        assert self.dim % 2 ==0, 'SinusoidalPosEmb requires "dim" to be even' # Even numbers of sin and cos freqs needed, hence even

    def forward(self, t:(torch.Tensor)) -> torch.Tensor:
        """
        The forward prop for sinusoidal embedding.

        Args:
            t (torch.Tensor): The time input, any shape that conforms to (B, ..., 1) is fine

        Returns:
            emb (torch.Tensor): The embedded output (B, dim)
        """
        t = t.flatten() # Flattenning the usual (B,1) -> (B) input of T
        half_dim = self.dim//2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = self.scale * t.unsqueeze(1) * emb.unsqueeze(0) # [Batch, 1] * [1, Half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb # Gets [Batch, Features (sin and cos)]

class TimestepEmb(nn.Module):
    """
    Timestep Embedder. Embeds the time given some kind of condition as with (B, in_dim) -> (B, out_dim)

    Args:
        in_dim (int): the outs dim taken from SinusoidalPosEmb
        time_embed_dim (int): the size of 1st layer
        act_fn (str): act fn of 1st layer
        out_dim (int): the size of 2nd layer
        post_act_fn (Optional, str): act fn of last layer
        cond_proj_dim (int): dim of conditional inputs (abundances), =3 for the entire project
    """  
    def __init__(
            self, 
            in_dim:(int)=128, 
            time_embed_dim:(int)=512, 
            act_fn:(str)='silu',
            out_dim:(int)=128, 
            post_act_fn:Optional[str]=None, 
            cond_proj_dim:(int)=3
            ):
        super().__init__()

        # Get vals
        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

        # Build layers
        self.linear_1 = nn.Linear(in_dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_dim)
        else:
            self.cond_proj = None
        
    def forward(self, sample:(torch.Tensor), condition:(torch.Tensor)) -> torch.Tensor:
        """
        Forward props the time embedding, given some kind of a condition. Projects the conditions to the same dim
        as the time embedding, adds them, and then returns the final embedding using an MLP all throughout.

        Args:
            sample (torch.Tensor): An embedded time tensor with shape (B, in_dim)
            condition (torch.Tensor): An abundance, condition tensor with any shape that adheres to (B, ..., 1)
        
        Returns:
            sample (torch.Tensor): Conditionally embedded time with dims (B, out_dim)
        """

        condition = condition.reshape(condition.shape[0], -1) # This ensures any kind of ab input like (B,1,3) or (B,1,..,1,3)

        # If using conditions, pass them into the condition projection, and add them to the input sample
        if condition is not None:
            sample = sample + self.cond_proj(condition)

        # Pass the sample through the first layer
        sample = self.linear_1(sample)

        # If using act after first hidden layer, pass it into act
        if self.act is not None:
            sample = self.act(sample)

        # Pass the sample through the second layer
        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)

        return sample 

class Downsample1D(nn.Module):
    """
    Downsampling layer. Essentially compresses the input sequence. Performs: (B, in_ch, seq_len) -> (B, out_ch, seq_len / 2)
    Usually should keep in and out channels the same, we only want to use this to downsample the sequence.

    Args:
        in_channels (int): The input channels before downsampling
        out_channels (int): The output channels after downsampling
    """
    def __init__(self, in_channels:(int), out_channels:(int)=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.conv = nn.Conv1d(self.in_channels, self.out_channels, 3, 2, 1)

    def forward(self, x:(torch.Tensor)) -> torch.Tensor: 
        """
        Forward propagation for the downsampling layer. Performs: (B, in_ch, seq_len) -> (B, out_ch, seq_len / 2)

        Args:
            x (torch.Tensor): The sample with shape (B, in_ch, seq_len)
        
        Returns:
            x (torch.Tensor): The sample with downsampled shape (B, out_ch, seq_len / 2)
        """
        return self.conv(x)
    
class Upsample1D(nn.Module):
    """
    Upsampling layer. Essentially decompresses the input sequence. Performs (B, in_ch, seq_len) -> (B, out_ch, seq_len * 2)
    Usually should keep in and out channels the same, we only want to use this to upsample the sequence.

    Args:
        in_channels (int): The input channels before upsampling
        use_conv_transpose (bool): Preference for ConvTranspose or Upsample+Conv
        out_channels (int): The output channels after upsampling
    """
    def __init__(self, in_channels:(int), use_conv_transpose:(bool)=True, out_channels:(int)=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        if use_conv_transpose:
            self.block = nn.ConvTranspose1d(self.in_channels, self.out_channels, 4, 2, 1)

        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv1d(self.in_channels, self.out_channels, 3, padding=1)
                )

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        """
        Forward propagation for the upsampling layer. Performs: (B, in_ch, seq_len) -> (B, out_ch, seq_len * 2)

        Args:
            x (torch.Tensor): The sample with shape (B, in_ch, seq_len)
        
        Returns:
            x (torch.Tensor): The sample with upsampled shape (B, out_ch, seq_len * 2)
        """
        return self.block(x)

class Block1D(nn.Module):
    """
    Takes performs convolution, group norms, applies Mish(). 

    Args:
        in_channels (int): Input channels of the 1D Block
        out_channels (int): Output channels of the 1D Block
        groups (int): how many groups to be seperated into, must divide out_channels
    """
    def __init__(self, in_channels:(int), out_channels:(int), groups:(int)=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1), # Just mixes channels: (B, in_ch, seq_len) -> (B, out_ch, seq_len)
            nn.GroupNorm(groups, out_channels), # Normalizes the statistics of (B, out_ch, seq_len) for each (B, group, seq_len) i.e. a mini-batch
            nn.Mish() # Applies non-linearity, elevent-wise Mish
        )

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        """
        Forward prop for a single 1D Block. Performs channel mixing using convolution, applies group norm to normalize the stats for each mini-batch defined
        by groups, and finally applies element-wise Mish.

        Args:
            x (torch.Tensor): The input with shape (B, in_ch, seq_len)

        Returns:
            x (torch.Tensor: Output sequence with shape (B, out_ch, seq_len) with operation Mish(GroupNorm(Conv1D))
        """
        return self.block(x)

class ResnetBlock1D(nn.Module):
    """
    An entire 1D Resnet block. Takes in the data of shape (B, in_ch, seq_len) -> (B, out_ch, seq_len). Effectively does Block1D((MLP(time_emb) + Block1D(x)) + Map(in->out(channels))

    Args:
        in_channels (int): The amount of input channels coming into the block
        out_channels (int): The amount of output channels coming out of the block
        time_emb_dim (int): The dimensions of time embeddings
        groups (int): Groups for group norm in Block1Ds
    """
    def __init__(self, in_channels:(int), out_channels:(int), time_emb_dim:(int), groups:(int)=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, out_channels)
        ) # The mlp that does: (B, time_emb_dim) -> Mish(B, time_emb_dim) -> (B, out_ch) this will be unsqueezed to broadcast over some values, to condition on time and abundances

        self.block1 = Block1D(in_channels, out_channels, groups) # Will do: (B, in_ch, seq_len) -> (B, out_ch, seq_len) and expands+mixes channels
        self.block2 = Block1D(out_channels, out_channels, groups) # Will do: (B, out_ch, seq_len) -> (B, out_ch, seq_len) and mixes channels

        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, 1) # If in_ch and out_ch do not match, residual connect the input by projecting over channels (B, in_ch, seq_len) -> (B, out_ch, seq_len)
        else:
            self.projection = nn.Identity() # If they do match, don't do anything to not waste params

    def forward(self, x:(torch.Tensor), time_emb:(torch.Tensor)) -> torch.Tensor:
        """
        Forward propagation for the 1D ResnetBlock, allows for a residual connection to avoid vanishing grads, get affected by the time_emb.

        Logic:
            h = Block1(x) + MLP(time_emb)
            y = Block2(h) + Projection(x)

        Args:
            x (torch.Tensor): The input sample of shape (B, in_ch, seq_len)
            time_emb (torch.Tensor): The time embedding of shape (B, time_emb_dim)

        Returns:
            y (torch.Tensor): Sample of shape (B, out_ch, seq_len)
        """
        h = self.block1(x) # Pass x through the first 1D Block: (B, in_ch, seq_len) -> (B, out_ch, seq_len) 

        time_h = (self.mlp(time_emb)).unsqueeze(-1) # Does (B, time_emb_dim) -> (B, out_ch) -> (B, out_ch, 1) which makes it broadcastable to output of 1st Block (B, out_ch, seq_len) 
        h += time_h # Broadcast the channel modified time embedding over the output of 1st 1D Block: (B, out_ch, seq_len) + (B, out_ch, 1) = (B, out_ch, seq_len)

        h = self.block2(h) # Pass it through to the second 1D Block (B, out_ch, seq_len) -> (B, out_ch, seq_len)

        return h + self.projection(x) # Add the residual connection from the original input, get (B, out_ch, seq_len)

class Epsilon_Cond1DUnet(nn.Module):
    """
    A conditional 1D U-Net that employs various transformer or conformer blocks and conducts convolution. For more detail about what it looks like, check page 3 of https://arxiv.org/pdf/2309.03199. 

    Args:
        n_bands (int): Also widely referred to as seq_len in the code. The total number of bands of the training dataset, not used in this class, but used in the DDPM class
        in_channel (int): How many channels are being used as ins (for example, a channel is pure spectra as a channel dim, or derivs of spectra as a channel dim distinctly different than seq_len/n_bands)
        out_channel (int): How many channels are being used for outs (usually 1)
        n_endmembers (int): Defaults to 3
        channels (list[int]): For up and down sections, how many 'large blocks' (that has a resnet layer and however many transformer/conformers) there will be to conduct expansion and shrinking
        n_former_blocks (int): How many (trans/con)former blocks to have at each 'large block' where channel amount is preserved
        n_mid_blocks (int): Total amount of mid blocks
        dropout (float): The dropout percentage throughout all the transformers/conformers
        dim_head (int): Dimensions of each head at transformers/conformers
        down/mid/up_type (str): Either 'conformer' or 'transformer'
        conv_kernel_size (int): Specifically needed for conformer types, kernel size used during convolutional layers
        conv_expansion_factor (int): Specifically needed for conformer layers, the size of channel expansion that internally happens in the conformer
        time_embed_dim (int): The dimension of time embeddings
    """
    def __init__(
            self,
            n_bands:(int),
            in_channel:(int)=1,
            out_channel:(int)=1,
            n_endmembers:(int)=3,
            channels:(list[int])=[64,128,256],
            n_former_blocks:(int)=1,
            n_mid_blocks:(int)=1,
            dropout:(float)=0.05,
            dim_head:(int)=64,
            down_type:(str)='conformer',
            mid_type:(str)='conformer',
            up_type:(str)='conformer',
            conv_kernel_size:(int)=9,
            conv_expansion_factor:(int)=2,
            time_embed_dim:(int)=128,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.target_out_channel = out_channel or in_channel # Take in either the out_channel if provided or in_channel

        block_cfg = {
            'dim_head': dim_head,
            'dropout': dropout,
            'conv_kernel_size': conv_kernel_size,
            'conv_expansion_factor': conv_expansion_factor
        } # Build the transformer/conformer block cfg for easier inputs

        self.down_factor = 2 ** (len(channels) - 1) # How many times we will scale down our sequence lengths in total. Will use this to pad the input tensor.
                                                    # We have 3 length modes seq_len, seq_len/2, seq_len/4 for def channel list [64, 128, 256]

        # Time embedding
        self.sin_emb = SinusoidalPosEmb(time_embed_dim) # This will take time, (B, ..., 1) -> (B, time_embed_dim) and perform a sinusoidal embedding
        self.time_emb = TimestepEmb(in_dim=time_embed_dim, out_dim=time_embed_dim) # This will take the sinusoidal embedding (B, time_embed_dim) -> (B, time_embed_dim)
                                                                                   # And embed them again depending on the abundances

        #region Block Building, (B, in_ch + n_endmembers, seq_len) -> (B, target_out_channel, seq_len)

        # Block construction, modulelists for the down_blocks, mid_blocks, and up_blocks sections
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        current_input_ch = in_channel + n_endmembers # Since we will broadcast the entire abundance vector (B, n_endmember) over the (B, 1, seq_len) 
                                                     # (assuming in_ch = 1). To get  (B, n_endmember + 1, seq_len), we have to keep track of 'n_endmember + 1'

        #region Down Loop, (B, 4, seq_len) -> (B, 256, seq_len/4)

        # The first down channel
        resnet = ResnetBlock1D(
            current_input_ch, channels[0], time_embed_dim
        ) # This first block will do (B, 4, seq_len) -> (B, 64, seq_len)
        transformers = nn.ModuleList(
            [
                self.get_block(down_type, dim=channels[0], **block_cfg) for j in range(n_former_blocks)
            ]
        ) # The transformer will do (B, 64, seq_len) n_former_blocks times

        self.down_blocks.append(nn.ModuleList([
                resnet, transformers, nn.Identity()
            ]))
        
        # Will do in order:
        # (B, 64, seq_len) -> (B, 128, seq_len) -- n_former_blocks times --> (B, 128, seq_len) -> (B, 128, seq_len/2)
        # (B, 128, seq_len/2) --> (B, 256, seq_len/2) -- n_former_blocks times --> (B, 256, seq_len/2) -> (B, 256, seq_len/4) this will enter mid blocks
        for i in range(len(channels) - 1):

            resnet = ResnetBlock1D(
                channels[i], channels[i+1], time_embed_dim 
            ) # Does (B, channels[i], seq_len) -> (B, channels[i+1], seq_len)
            transformers = nn.ModuleList(
                [
                    self.get_block(down_type, dim=channels[i+1], **block_cfg) for j in range(n_former_blocks)
                ]
            ) # Does (B, channels[i+1], seq_len) -> (B, channels[i+1], seq_len). This happens n_former_blocks times
            downsample = Downsample1D(channels[i+1])
            self.down_blocks.append(nn.ModuleList([
                resnet, transformers, downsample
            ])) # Does (B, channels[i+1], seq_len/2*i) -> (B, channels[i+1], seq_len/2*i+1)
        #endregion

        #region Mid Loop, (B, 256, seq_len/4) -> (B, 256, seq_len/4)

        # Does: (B, 256, seq_len/4) -> (B, 256, seq_len/4) -- n_former_blocks times --> (B, 256, seq_len/4), the entire process n_mid_blocks times
        for _ in range(n_mid_blocks):
            resnet = ResnetBlock1D(
                channels[-1], channels[-1], time_embed_dim
            ) # Does (B, 256, seq_len/4) -> (B, 256, seq_len/4)
            transformers = nn.ModuleList(
                [
                    self.get_block(mid_type, dim=channels[-1], **block_cfg) for j in range(n_former_blocks)
                ]
            ) # Does (B, 256, seq_len/4) -> (B, 256, seq_len/4). This happens n_former_blocks times
            self.mid_blocks.append(nn.ModuleList(
                [
                    resnet, transformers
                ]
            ))
        #endregion

        #region Up Loop, (B, 512, seq_len/4) *concatted -> (B, target_out_channel, seq_len)

        reversed_channels = list(reversed(channels)) # Reverses channels so we can loop them reversed
        # Will do in order:
        # (B, 512, seq_len/4) *concatted -> (B, 128, seq_len/4) -- n_former_blocks times --> (B, 128, seq_len/4) -> (B, 128, seq_len/2)
        # (B, 256, seq_len/2) *concatted -> (B, 64, seq_len/2) -- n_former_blocks times --> (B, 64, seq_len/2) -> (B, 64, seq_len)
        for i in range(len(reversed_channels) - 1):
            resnet = ResnetBlock1D(
                reversed_channels[i]*2, reversed_channels[i+1], time_embed_dim 
            ) # (B, reversed_channels[i]*2, seq_len/div) *concatted -> (B, reversed_channels[i+1], seq_len/div)
            transformers = nn.ModuleList(
                [
                    self.get_block(up_type, reversed_channels[i+1], **block_cfg) for j in range(n_former_blocks)
                ]
            ) # (B, reversed_channels[i], seq_len/div) -- n_former_blocks times --> (B, reversed_channels[i], seq_len/div)
            upsample = Upsample1D(reversed_channels[i])
            self.up_blocks.append(nn.ModuleList([
                resnet, transformers, upsample
            ])) # (B, reversed_channels[i+1], seq_len*2/div) -> (B, reversed_channels[i+1], seq_len*2/div)

        self.final_block = Block1D(channels[0] * 2, channels[0]) # (B, 128, seq_len) *concatted -> (B, 64, seq_len)
        self.final_proj = nn.Conv1d(channels[0], self.target_out_channel, 1) # (B, 64, seq_len) -> (B, target_out_channel, seq_len)
        #endregion

        #endregion

    @staticmethod 
    def get_block(block_type:(str), dim:(int), **kwargs) -> Union[Transformer1D, Conformer1D]:  
        """
        This static method is to get the blocks as either conformer or transformer. Made dim as a standalone input because dim 
        is constantly changed which also affects total amount of heads in transformer/conformer. **kwargs are assumed to be 
        dim_head, dropout, conv_kernel_size, conv_expansion_factor.

        Args:
            block_type (str): Either 'transformer' or 'conformer'
            dim (int): The amount of channel dims for the layer
            dim_head (int): Dim per head
            dropout (float): Dropout amount for the layer
            conv_kernel_size (int): Used if block_type is conformer, the kernel size for internal convolutions in conformer
            conv_expansion_factor (int): Used if block_type is conformer, it is how much channel expansion happens internally (output is still same ch)

        Returns:
            block (nn.Module): Either a Transformer1D or Conformer1D class    
        """
        # **kwargs has: dim_head, dropout, conv_kernel_size, 
        if block_type == 'transformer': 
            valid_args = {'dim_head', 'dropout'} # Transformer expects: dim, dim_head, dropout
            kwargs = {k: v for k, v in kwargs.items() if k in valid_args}  # Filter the kwargs
            return Transformer1D(dim, **kwargs) # Discards conv_kernel_size, conv_expansion_factor in the **kwargs dict
        elif block_type == 'conformer': 
            # Conformer expects: dim, dim_head, dropout, conv_kernel_sze, conv_expansion_factor so, we pass all in
            return Conformer1D(dim, **kwargs) # Takes in all the **kwargs
        else:
            raise ValueError(f"Unexpected block type: {block_type}")

    def forward(self, x_t:(torch.Tensor), t:(torch.Tensor), ab:(torch.Tensor)) -> torch.Tensor:
        """
        The forward prop for the epsilon, requires:

        Args:
            x_t (torch.Tensor): The noised sample at time t with shape (B, seq_len) or (B, i, seq_len)
            t (torch.Tensor): Timestep t
            ab (torch.Tensor): The abundances of the noised sample
        
        Returns:
            x_(t-1) (torch.Tensor): The noise removed, at timestep t-1
        """
        
        # Transformations on x to ensure compatibility
        if x_t.ndim == 2: # Usually, our inputs are (B, seq_len), we need to add channel dim
            x_t = x_t.unsqueeze(1) # This gives (B, 1, seq_len), this will be overridden if giving derivatives of spectra as another channel: (B, 2, seq_len)
        assert 1 < x_t.ndim < 4, "Broski I have no idea why you'd input a 1 dim or 4+ dim shit to this"
        n_bands = x_t.shape[-1]
        remainder = n_bands % self.down_factor # Figure out how much padding we gotta do to not get seq_len like 40.5
        if remainder != 0:
            # Pad the right side so the length is divisible by the down_factor
            pad_len = self.down_factor - remainder
            x_t = F.pad(x_t, (0, pad_len)) # Ensures that throughout the operations, (B, ch, seq_len) seq_len is an int
        # Ensure the ab tensor
        assert ab.ndim == 2, f"I have no idea why you'd input the ndim of abundances as {ab.ndim} and not 2"

        t = self.sin_emb(t) # Embed the time sinusoidally (B, ..., dim) -> (B, time_embed_dim)
        t = self.time_emb(t, ab) # Embed the time again now using abundances as ins as well (B, time_embed_dim) -> (B, time_embed_dim)

        ab_expanded = repeat(ab, 'b c -> b c t', t=x_t.shape[-1]) # Unpack the abundance vector to get (B, n_endmember) -> (B, n_endmember, seq_len) it becomes broadcastable
        x_t = pack([x_t, ab_expanded], 'b * t')[0] # Append the entire abundance tensor over the spectral tensor: (B, in_ch, seq_len) + (B, n_endmember, seq_len) -> (B, in_ch + n_endmember, seq_len)

        hiddens=[] # Create a list to store all the hiddens

        # For the default setup, it does (B, in_ch + n_endmembers, seq_len) -> (B, 256, seq_len/4)
        for resnet, transformers, downsample in self.down_blocks:
            x_t = resnet(x_t, t) # First do the resnet to transform (B, ch, seq_len) -> (B, 2*ch, seq_len)
            for block in transformers: x_t = block(x_t) # Do global mixing using transformer n_former_blocks times: (B, 2*ch, seq_len) -> (B, 2*ch, seq_len)
            hiddens.append(x_t) # Append the hidden result
            x_t = downsample(x_t) # Downsample to end up with (B, 2*ch, seq_len/2). Note, this is identity for the first downsample

        # Constantly does (B, 256, seq_len/4) -> (B, 256, seq_len/4) over and over, n_mid_blocks times
        for resnet, transformers in self.mid_blocks: 
            x_t = resnet(x_t, t)
            for block in transformers: x_t = block(x_t)

        # For the default setup, it does (B, 256, seq_len/4) -> (B, 64, seq_len)
        for resnet, transformers, upsample in self.up_blocks: # Start with (B, )
            skip = hiddens.pop() # The hiddens have size (B, ch, seq_len)
            x_t = upsample(x_t) # Upsample to get (B, ch, seq_len/2) -> (B, ch, seq_len)
            x_t = pack([x_t, skip], 'b * t')[0] # Pack them to get (B, 2*ch, seq_len)
            x_t = resnet(x_t, t) # Perform (B, 2*ch, seq_len) -> (B, ch, seq_len)
            for block in transformers: x_t = block(x_t) # Do global mixing using transformer n_former_blocks times: (B, ch, seq_len) -> (B, ch, seq_len)
        # End up with (B, 64, seq_len)

        skip = hiddens.pop() # Take the last hidden, (B, 64, seq_len)
        x_t = pack([x_t, skip], 'b * t')[0] # Concat out of up blocks with last hidden: (B, 128, seq_len)

        x_t = self.final_block(x_t) # Project (B, 128, seq_len) -> (B, 64, seq_len) using a 1D Block
        x_t = self.final_proj(x_t) # Finally project (B, 64, seq_len) -> (B, target_out_ch, seq_len)
        x_t = x_t[..., :n_bands] # Slice the x_t tensor since we have padded it already to ensure divisibility, get (B, target_out_ch, n_bands)
        if self.target_out_channel == 1: x_t=x_t.squeeze(1) # Squeeze the channel dim if we have out channel dim as 1
        return x_t





        