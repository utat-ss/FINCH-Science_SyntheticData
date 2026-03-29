from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer.

    Input:
      x: (batch, in_channels, B)

    Steps:
      1) rFFT along B -> (batch, in_channels, B_fft)
      2) keep low modes
      3) multiply by learned complex weights
      4) iFFT back to length B -> (batch, out_channels, B)

    Output:
      (batch, out_channels, B)  
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # complex weights stored as (..., 2) for real/imag
        # shape: (in_channels, out_channels, modes, 2)
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x=(batch,in_ch,B), got {tuple(x.shape)}")

        b, in_ch, B = x.shape
        if in_ch != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {in_ch}")

        # (batch, in_ch, B_fft)
        x_ft = torch.fft.rfft(x, dim=-1)
        B_fft = x_ft.size(-1)

        m = min(self.modes, B_fft)

        # (batch, out_ch, B_fft)
        out_ft = torch.zeros(b, self.out_channels, B_fft, device=x.device, dtype=torch.cfloat)

        # (in_ch, out_ch, m) complex
        w = torch.view_as_complex(self.weight[:, :, :m, :].contiguous())

        # low-mode mixing:
        # x_ft[:, :, :m] -> (b, in_ch, m)
        # out_ft[:, :, :m] -> (b, out_ch, m)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], w)

        # back to (batch, out_ch, B)   
        x_out = torch.fft.irfft(out_ft, n=B, dim=-1)
        return x_out


class FNO1DUnmixing(nn.Module):
    """
    Per-pixel hyperspectral unmixing with FNO1D.

    Input:
      spectra: (batch, 1, B)   (we will unsqueeze in training)

    Output:
      abundances: (batch, K)   (softmax enforced)
    """

    def __init__(
        self,
        num_endmembers: int,
        modes: int = 32,
        width: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        if pool not in {"mean", "max"}:
            raise ValueError("pool must be 'mean' or 'max'")

        self.num_endmembers = num_endmembers
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.pool = pool

        # lift per band: (batch, B, 1) -> (batch, B, width)
        self.fc0 = nn.Linear(1, width)

        self.spectral_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(num_layers)])

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # project: (batch, width) -> (batch, K)
        self.fc1 = nn.Linear(width, num_endmembers)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        spectra: (batch, 1, B)
        """
        if spectra.ndim != 3:
            raise ValueError(f"Expected spectra=(batch,1,B), got {tuple(spectra.shape)}")
        b, c, B = spectra.shape
        if c != 1:
            raise ValueError(f"Expected channel=1, got {c}")

        x = spectra
        # (batch, 1, B) -> (batch, B, 1)
        x = x.permute(0, 2, 1)

        # lift: (batch, B, 1) -> (batch, B, width)
        x = self.fc0(x)

        # to conv format: (batch, width, B)
        x = x.permute(0, 2, 1)

        # Fourier layers
        for spec_conv, w in zip(self.spectral_convs, self.ws):
            # spec_conv(x): (batch, width, B)
            # w(x):         (batch, width, B)
            y = spec_conv(x) + w(x)
            x = x + self.drop(self.act(y))  # residual + activation

        # pool over spectral dimension -> (batch, width)
        if self.pool == "mean":
            x = x.mean(dim=-1)
        else:
            x = x.max(dim=-1).values

        logits = self.fc1(x)           # (batch, K)
        abund = F.softmax(logits, dim=1)
        return abund
