"""
U-Net Architecture for Acoustic Refinement

Standard U-Net with:
- 4 encoder blocks with downsampling
- Bottleneck
- 4 decoder blocks with upsampling
- Skip connections
- Timestep conditioning via sinusoidal embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings for timestep conditioning.
    
    Similar to positional encodings in transformers, allows the network
    to distinguish between different refinement iterations.
    """
    
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timestep indices [1, 2, 3, 4]
            
        Returns:
            (B, dim) tensor of sinusoidal embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Create frequency scale
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Compute embeddings
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with batch norm and ReLU.
    Standard U-Net building block.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block with maxpool followed by double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsampling block with transposed convolution followed by double conv.
    Includes skip connection concatenation.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        
        # Double conv after concatenation with skip connection
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from previous layer
            skip: Skip connection from encoder
        """
        x = self.up(x)
        
        # Handle size mismatch due to pooling
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for spectrogram refinement with timestep conditioning.
    
    Architecture:
    - Input: (B, 1, 80, T) spectrogram + timestep index
    - Encoder: 4 downsampling blocks (64 → 128 → 256 → 512 → 1024 channels)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks (1024 → 512 → 256 → 128 → 64)
    - Output: (B, 1, 80, T) predicted noise residual
    
    Args:
        in_channels: Input channels (default: 1 for single-channel spectrogram)
        out_channels: Output channels (default: 1)
        base_channels: Base number of channels (default: 64)
        timestep_dim: Dimension of timestep embeddings (default: 128)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        timestep_dim: int = 128
    ):
        super().__init__()
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(timestep_dim)
        
        # MLP to project timestep embedding to spatial dimensions
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_dim, base_channels * 16),
            nn.ReLU(),
            nn.Linear(base_channels * 16, base_channels * 16)
        )
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Encoder (downsampling path)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 16, base_channels * 16)
        
        # Decoder (upsampling path)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output projection
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"U-Net initialized with {total_params:,} parameters")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram (B, 1, 80, T)
            t: Timestep indices (B,) in range [1, 2, 3, 4]
            
        Returns:
            Predicted noise residual (B, 1, 80, T)
        """
        # Embed timestep
        t_emb = self.time_embed(t)  # (B, timestep_dim)
        t_emb = self.time_mlp(t_emb)  # (B, base_channels * 4)
        
        # Reshape for broadcasting
        # (B, C, 1, 1) to add to spatial feature maps
        t_emb = t_emb[:, :, None, None]
        
        # Encoder path with skip connections
        x1 = self.inc(x)  # (B, 64, 80, T)
        
        x2 = self.down1(x1)  # (B, 128, 40, T/2)
        x3 = self.down2(x2)  # (B, 256, 20, T/4)
        x4 = self.down3(x3)  # (B, 512, 10, T/8)
        x5 = self.down4(x4)  # (B, 1024, 5, T/16)
        
        # Bottleneck with timestep conditioning
        x = self.bottleneck(x5)
        x = x + t_emb  # Add timestep information
        
        # Decoder path with skip connections
        x = self.up1(x, x4)  # (B, 512, 10, T/8)
        x = self.up2(x, x3)  # (B, 256, 20, T/4)
        x = self.up3(x, x2)  # (B, 128, 40, T/2)
        x = self.up4(x, x1)  # (B, 64, 80, T)
        
        # Output projection
        noise_pred = self.outc(x)  # (B, 1, 80, T)
        
        return noise_pred


def test_unet():
    """Test U-Net forward pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet().to(device)
    
    # Create dummy input
    batch_size = 2
    mel_channels = 80
    time_frames = 1500  # ~30 seconds of audio
    
    x = torch.randn(batch_size, 1, mel_channels, time_frames).to(device)
    t = torch.randint(1, 5, (batch_size,)).to(device)  # Timesteps 1-4
    
    # Forward pass
    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {t}")
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with different time frames
    for frames in [500, 1000, 1500, 2000]:
        x_test = torch.randn(1, 1, mel_channels, frames).to(device)
        t_test = torch.tensor([2]).to(device)
        
        with torch.no_grad():
            out_test = model(x_test, t_test)
        
        print(f"\nFrames: {frames} → Output: {out_test.shape}")


if __name__ == "__main__":
    test_unet()
