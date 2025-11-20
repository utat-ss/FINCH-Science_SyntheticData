import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.activations import get_activation

from defs.diffusion.epsilon.auxiliary.epsilon_addons import Transformer1D, Conformer1D

import math

from typing import Optional
from einops import repeat, pack

"""
The grand majority of this code was taken from: @inproceedings{mehta2024matcha,title={Matcha-{TTS}: A fast {TTS} 
architecture with conditional flow matching}, author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, 
{\'E}va and Henter, Gustav Eje}, booktitle={Proc. ICASSP},year={2024}}

Primary and the most important changes are:
1- Removal of the mask option, since we assume a constant spectral band amount (~210)
2- Removal of the mu conditioning input, and its change to abundances
3- Addition of comments, improved readability
4- Simplification of some code and removal of boilerplate code
"""

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim:(int)=128):
        super().__init__()

        self.dim = dim
        assert self.dim % 2 ==0, 'SinusoidalPosEmb requires "dim" to be even' # Even numbers of sin and cos freqs needed, hence even

    def forward(self, x, scale:(int)=1000):
        
        if x.ndim < 1: # If x is a straight up float, make it a single-entry vector
            x=x.unsqueeze(0)

        half_dim = self.dim//2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0) # [Batch, 1] * [1, Half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb # Gets [Batch, Features (sin and cos)]
    
class TimestepEmb(nn.Module):
    
    """
    Timestep Embedder

    Args:
        in_channels: the outs' dim taken from SinusoidalPosEmb
        time_embed_dim: the size of 1st layer
        act_fn: act fn of 1st layer
        out_dim: the size of 2nd layer
        post_act_fn: act fn of last layer
        cond_proj_dim: dim of conditional inputs (abundances), =3 for the entire project

    Returns:
        sample: embedded time samples
    """
    
    def __init__(
            self, 
            in_channels:(int)=128, 
            time_embed_dim:(int)=512, 
            act_fn:(str)='silu',
            out_dim:(int)=512, 
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

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels)
        else:
            self.cond_proj = None
        
        def forward(self, sample, condition):

            # If using conditions, pass them into the condition projection, and add them to the input sample
            if condition is not None:
                sample = sample + self.cond_proj(condition)

            # Pass the sample through the first layer
            sample = self.linear_1(sample)

            # If using act after first hidden layer, pass it into act
            if self.act is not None:
                sample = self.act(sample)

            # Pass the sample through the second layer
            if self.post_act is not None:
                sample = self.post_act(sample)

            return sample 

class Downsample1D(nn.Module):

    """
    Downsampling layer.

    In and out channels usually need to be equal.
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.conv = nn.Conv1d(self.in_channels, self.out_channels, 3, 2, 1)

    def forward(self, x):

        return self.conv(x)
    
class Upsample1D(nn.Module):

    """
    Upsampling layer.

    In and out channels usually need to be equal. If use_conf_transpose=False, uses interpolation and convolution.
    """
    
    def __init__(self, in_channels, use_conv_transpose:(bool)=True, out_channels=None):
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

    def forward(self, x):

        return self.block(x)

class Block1D(nn.Module):

    """
    Takes performs convolution, group norms, applies Mish(). 

    Args:
        in_channels: input channels
        out_channels: output channels
        groups: how many groups to be seperated into, must divide out_channels

    Returns:
        Mish(GroupNorm(Conv1D))
    """

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.Mish()
        )

    def forward(self, x):

        return self.block(x)

class ResnetBlock1D(nn.Module):

    """
    An entire 1D Resnet block. 

    Args:
        in_channels: The amount of input channels coming into the block
        out_channels: The amount of output channels coming out of the block
        
    Returns:
        Block1D( MLP(time_emb) + Block1D(x) ) + Map(in->out(channels))
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.block1 = Block1D(in_channels, out_channels, groups)
        self.block2 = Block1D(out_channels, out_channels, groups)

        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conb = nn.Identity()

    def forward(self, x, time_emb):

        h = self.block1(x)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h)
        
        return h + self.res_conv(x)

class Decoder: # Equivalent to epsilon_cond1DUnet


    pass

class Epsilon_Cond1DUnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        in_channel = kwargs['in_channels']
        out_channel = kwargs['out_channel']
        n_endmembers = kwargs.get('n_endmembers', 3)

        channels = kwargs.get('channels', [64,128])
        n_blocks = kwargs.get('n_blocks', 1)
        dropout = kwargs.get('dropout', 0.05)
        num_heads = kwargs.get('num_heads', 4)
        head_dim = kwargs.get('head_dim', 64)
        kernel_size = kwargs.get('kernel_size', 9)

        down_type = kwargs.get('down_type', 'conformer')
        mid_type = kwargs.get('mid_tyoe', 'conformer')
        up_type = kwargs.get('up_type', 'conformer')

        block_cfg = {
            'num_heads': num_heads,
            'dropout': dropout,
            'kernel_size': kernel_size,
            'head_dim': 64
        }

        # Time embedding

        time_embed_dim = 128

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim), TimestepEmb(time_embed_dim)
        )


        # Block construction

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        current_input_dim = in_channel + n_endmembers
        self.output_channel = channels[0]

        # Down Loop

        for i in range(len(channels)):

            out_channel = channels[i]

            is_last = i == len(channels) - 1

            resnet = ResnetBlock1D(
                current_input_dim, out_channel, time_embed_dim 
            )

            transformers = nn.ModuleList(
                [
                    self.get_block(down_type, out_channel, **block_cfg) for i in range(n_blocks)
                ]
            )

            downsample = Downsample1D(out_channel) if not is_last else nn.Conv1d(out_channel, out_channel, 3, padding=1)

            self.down_blocks.append(nn.ModuleList([
                resnet, transformers, downsample
            ]))
            current_input_dim = out_channel

        
        # Mid Loop

        resnet = ResnetBlock1D(
            current_input_dim, current_input_dim, time_embed_dim
        )

        transformers = nn.ModuleList(
            [
                self.get_block(mid_type, current_input_dim, **block_cfg)
            ]
        )

        self.mid_blocks.append(nn.ModuleList[
            resnet, transformers
        ])


        # Up Loop

        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels) - 1):

            in_channel = reversed_channels[i]
            out_channel = reversed_channels[i+1]

            resnet = ResnetBlock1D(
                in_channel*2, out_channel, time_embed_dim
            )

            transformers = nn.ModuleList(
                [
                    self.get_block(up_type, out_channel, **block_cfg)
                ]
            )

            upsample = Upsample1D(out_channel)

            self.up_blocks.append(nn.ModuleList([
                resnet, transformers, upsample
            ]))
        

        # Final 

        self.final_block = Block1D(channels[0], channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channel, 1)

    @staticmethod
    def get_block(block_type, dim, **kwargs):

        if block_type == 'transformer':
            return Transformer1D(**kwargs)
        elif block_type == 'conformer':
            return Conformer1D(**kwargs)
        else:
            raise ValueError(f"Unexpected block type: {block_type}")

    def forward(self, x_t, t, ab):

        t = self.time_embedding(t)

        ab_expanded = repeat(ab, 'b c -> b c t', t=x_t.shape[-1])
        x_t = pack([x_t, ab_expanded], 'b * t')[0]

        hiddens=[]

        for resnet, transformers, downsample in self.down_blocks:
            x_t = resnet(x_t, t)
            for block in transformers: x_t = block(x_t)
            hiddens.append(x_t)
            x_t = downsample(x_t)

        for resnet, transformers in self.mid_blocks:
            x_t = resnet(x_t, t)
            for block in transformers: x_t = block(x_t)

        for resnet, transformers, upsample in self.up_blocks:
            skip = hiddens.pop()
            x_t = pack([x_t, skip], 'b * t')[0]
            x_t = resnet(x_t)
            for block in transformers: x_t = block(x_t)
            x_t = upsample(x_t)

        x_t = self.final_block(x_t)
        return self.final_proj(x_t)





        