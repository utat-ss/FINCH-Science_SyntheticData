import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1d(nn.Module):
    """
    conv -> norm -> gelu -> dropout
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 5, 
            dropout: float = 0.0,
            norm: str = "batch" # "batch" | "group" | "none"
    ):
        super().__init__()

        # padding keeps the length the same (so B doesn't shrink after conv)
        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "group":
            groups = 8 if out_channels >= 8 else 1
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unknown norm: {norm}")
        
        self.act = nn.GELU()

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x
    
class ResidualBlock1d(nn.Module):
    """
    (block -> block) + skip
    """
    def __init__(
            self,
            channels: int,
            kernel_size: int = 5,
            dropout: float = 0.0,
            norm: str = "batch"
    ):
        super().__init__()
        self.b1 = ConvBlock1d(channels, channels, kernel_size, dropout, norm)
        self.b2 = ConvBlock1d(channels, channels, kernel_size, dropout, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b2(self.b1(x))
    
class CNNUnmixing1D(nn.Module):
    """
    1D CNN over wavelength axis.
    input:  (batch, 1, B)
    output: (batch, K) with softmax (simplex)
    """
    def __init__(
            self,
            num_bands: int,
            num_endmembers: int = 3,
            channels: list[int] = None,
            kernel_size: int = 5,
            dropout: float = 0.0,
            norm: str = "batch",
            residual: bool = True,
            res_blocks_per_stage: int = 1,
            pool: str = "avg" # "avg" | "max"
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        self.num_bands = num_bands
        self.num_endmembers = num_endmembers

        layers = []
        in_ch = 1 # because input is (batch, 1, B)
        for ch in channels:
            # widen feature channels
            layers.append(ConvBlock1d(in_ch, ch, kernel_size, dropout, norm))

            # optional residual refinement at same width
            if residual:
                for _ in range(res_blocks_per_stage):
                    layers.append(ResidualBlock1d(ch, kernel_size, dropout, norm))
            
            # downsample wavelength axis to build hierarchy
            if pool == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif pool == "max":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"unknown pool: {pool}")
            
            in_ch = ch
        
        self.backbone = nn.Sequential(*layers)

        # collapses wavelength axis to length 1: (batch, C, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # small head: turn (batch, C, 1) -> (batch, K)
        hidden = max(in_ch // 2, num_endmembers * 4)
        self.head = nn.Sequential(
            nn.Flatten(),                 # (batch, C)
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, num_endmembers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, B)
        x = self.backbone(x)   # (batch, C, B')
        x = self.gap(x)        # (batch, C, 1)
        logits = self.head(x)  # (batch, K)

        # simplex constraint (non-neg + sums to 1)
        return F.softmax(logits, dim=-1)