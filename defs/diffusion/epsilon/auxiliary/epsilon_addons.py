import torch.nn as nn

from einops import rearrange

from conformer import ConformerBlock

class FeedForward(nn.Module):

    """
    Simple feedforward layer.

    Args:
        i_dim: The input dimension
        mult: multiplication amount for the hidden dim wrt i_dim
        dropout: amount of dropout
    
    Returns:
        Dropout(Linear(Dropout(GELU(Linear(x)))))
    """

    def __init__(self, i_dim, mult=4, dropout=0.0):
        super().__init__()

        inner_dim = int(i_dim * mult)

        self.net = nn.Sequential(
            nn.Linear(i_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, i_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer1D(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=64, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Classic Self-Attention, Multi-Headed

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            batch_first=True
        )

        self.feedforward = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, x, **kwargs):

        """
        x: [Batch, Channels, Time]
        Catches and ignores 'timestep' and 'mask'
        """

        h = rearrange(x, 'b c t -> b t c' ) # Rearrange needed because this is what transformers assume

        h_norm = self.norm1(h)
        attn_out, _ = self.attention(h_norm, h_norm, h_norm) # Don't need weights output
        h += attn_out

        h += self.feedforward(self.norm2(h))

        return rearrange(h, 'b t c -> b c t')

class Conformer1D(nn.Module):

    """
    A conformal 1D block
    """

    def __init__(self, dim, num_heads=4, dropout=0.0, conv_kernel_size=9, conv_expansion_factor=2):
        super().__init__()

        self.block = ConformerBlock(
            dim=dim,
            dim_head= dim//num_heads,
            ff_mult= 4,
            conv_expansion_factor= conv_expansion_factor,
            conv_kernel_size= conv_kernel_size,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

    def forward(self, x, **kwargs):

        x = rearrange(x, 'b c t -> b t c')
        x = self.block(x)
        return rearrange(x, 'b t c -> b c t')
