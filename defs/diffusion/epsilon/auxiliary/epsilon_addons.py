import torch.nn as nn
import torch

from einops import rearrange

from conformer import ConformerBlock

class FeedForward(nn.Module):
    """
    Standard FeedForward layer. Projects into a higher hidden ch dim, applies non-linearity, projects back into the input ch dim size.
    Assumed (B, seq_len, ch) instead of (B, ch, seq_len) since that's what transformers assume.

    Args:
        i_ch_dim (int): The input channel dimension
        mult (float): Multiplication amount for the hidden dim i.e. hidden_dim = mult * i_dim
        dropout (float): Amount of dropout after each Sigma(Linear)
    """
    def __init__(self, i_ch_dim:(int), mult:(float)=4, dropout:(float)=0.0):
        super().__init__()

        inner_ch_dim = int(i_ch_dim * mult) # Multiply the i_ch_dim with mult to get the projection / inner_ch_dim

        self.net = nn.Sequential(
            nn.Linear(i_ch_dim, inner_ch_dim), # Linearly project (B, seq_len, in_ch_dim) -> (B, seq_len, inner_ch_dim)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_ch_dim, i_ch_dim), # Project back (B, seq_len, inner_ch_dim) -> (B, seq_len, in_ch_dim)
            nn.Dropout(dropout)
        )

    def forward(self, x:(torch.Tensor)) -> torch.Tensor:
        """
        Forward propagation for the FeedForward layer.

        Logic:
            x -> Linear(i_ch_dim, i_ch_dim*mult) -> GELU -> Dropout -> Linear(i_ch_dim*mult, i_ch_dim) -> Dropout
        
        Args:
            x (torch.Tensor): Input of shape (B, seq_len, in_ch_dim)

        Returns:
            y (torch.Tensor): Output of shape (B, seq_len, in_ch_dim)
        """
        return self.net(x)

class Transformer1D(nn.Module):
    """
    1D Transformer that applies LayerNorm, Attention, and a simple FeedForward. Assumes (B, ch, seq_len) as input shape

    Args:
        dim (int): The dim (amount) of the channels present i.e. (B; dim, seq_len)
        dim_head (int): Dimension of each head in the attention
        dropout (float): Dropout for the FeedForward layer
    """
    def __init__(self, dim:(int), dim_head:(int)=64, dropout:(float)=0.0):
        super().__init__()

        num_heads = dim // dim_head # Automatically set it such that num_heads always follows how many dims we have and how much dim we want for each head
        if dim % dim_head != 0:
            raise ValueError(f'dim: {dim} must be divisible by dim_head: {dim_head}')

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Classic Self-Attention, Multi-Headed

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Important, with this we have (B, seq_len, ch)
        )

        self.feedforward = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, x:(torch.Tensor), *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation for the 1D Transformer. 

        Logic:
            x = x + Attn(Norm1(x))
            y = FeedForward(Norm2(x))

        Args:
            x (torch.Tensor): Input tensor of the sample with shape (B, ch, seq_len)

        Returns:
            y (torch.Tensor): Output with the 1D Transformer applied with shape (B, ch, seq_len)
        """
        h = rearrange(x, 'b c t -> b t c' ) # Rearrange needed because this is what transformers assume compared to what our U-Net assumes

        h_norm = self.norm1(h) # Apply norm to the input of (B, seq_len, ch)
        attn_out, _ = self.attention(h_norm, h_norm, h_norm) # Applies attention to the normed input, only need the out
        h += attn_out # Add the Attn(Norm(x)) to x, this is widely used in modern models

        h += self.feedforward(self.norm2(h)) # FeedForward x + Attn(Norm(x)) with shape (B, seq_len, ch) to get the same shape

        return rearrange(h, 'b t c -> b c t') # Rearrange to conform with the shape assumptions of the U-Net

class Conformer1D(nn.Module):
    """
    A wrapped around the 1D Conformer. Expects input of shape (B, ch, seq_len) as assumed by the U-Net

    Args:
        dim (int): The dim (amount) of the channels present i.e. (B; dim, seq_len)
        dim_head (int): Dimension of each head in the attention
        dropout (float): Dropout for the FeedForward, Attention, and Convolution layers of the Conformer
        conv_kernel_size (int): The size of the kernels in the Conformer's convolutional layer
        conv_expansion_factor (int): The size of channel expansion that internally happens in the Conformer
    """
    def __init__(self, dim:(int), dim_head:(int)=64, dropout:(float)=0.0, conv_kernel_size:(int)=9, conv_expansion_factor:(int)=2):
        super().__init__()

        num_heads = dim // dim_head # Automatically set it such that num_heads always follows how many dims we have and how much dim we want for each head

        if dim % dim_head != 0:
            raise ValueError(f'dim: {dim} must be divisible by dim_head: {dim_head}')

        # For more detail on how to conformer works, check the details in its docs
        self.block = ConformerBlock(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            ff_mult=4,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        ) 

    def forward(self, x, *args, **kwargs):
        """
        Forward propagation for the 1D Conformer layer.

        Logic:
            y = Conformer1D(x)

        Args:
            x (torch.Tensor): Input of shape (B, ch, seq_len)

        Returns:
            y (torch.Tensor): Output of shape (B, ch, seq_len) with conformer applied
        """
        x = rearrange(x, 'b c t -> b t c')
        x = self.block(x) # Rearrange before and after to comply with U-Net and Conformer assumptions
        return rearrange(x, 'b t c -> b c t')
