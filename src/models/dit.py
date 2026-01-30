"""
Diffusion Transformer (DiT) models for stroke generation.
Includes both unconditional and image-conditioned variants.
"""
import torch
import torch.nn as nn
import math


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps using sinusoidal encoding + MLP."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (B,)
        Returns:
            Timestep embedding of shape (B, hidden_size)
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class ImageEncoder(nn.Module):
    """Encode images into conditioning vectors."""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, 128, 128)
        Returns:
            Image embedding of shape (B, output_dim)
        """
        return self.net(x)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (adaLN) modulation."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            c: Conditioning tensor of shape (B, D)
        Returns:
            Output tensor of shape (B, L, D)
        """
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        
        # MLP with modulation
        return x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class HeavyStrokeDiT(nn.Module):
    """
    Unconditional DiT for stroke generation.
    Takes noisy strokes + timestep, outputs denoised strokes.
    """
    
    def __init__(
        self,
        feature_dim: int = 65,
        seq_len: int = 20,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.x_embedder = nn.Linear(feature_dim, d_model)
        self.t_embedder = TimestepEmbedder(d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, feature_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy strokes of shape (B, seq_len, feature_dim)
            t: Timestep of shape (B,)
        Returns:
            Predicted clean strokes of shape (B, seq_len, feature_dim)
        """
        x = self.input_norm(self.x_embedder(x) + self.pos_embed)
        c = self.t_embedder(t)
        
        for block in self.blocks:
            x = block(x, c)
        
        return self.final_layer(x)


class VisualStrokeDiT(nn.Module):
    """
    Image-conditioned DiT for stroke generation.
    Takes noisy strokes + timestep + target image, outputs denoised strokes.
    """
    
    def __init__(
        self,
        feature_dim: int = 65,
        seq_len: int = 20,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.x_embedder = nn.Linear(feature_dim, d_model)
        self.t_embedder = TimestepEmbedder(d_model)
        self.img_encoder = ImageEncoder(output_dim=d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, feature_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy strokes of shape (B, seq_len, feature_dim)
            t: Timestep of shape (B,)
            img: Target image of shape (B, 3, 128, 128)
        Returns:
            Predicted clean strokes of shape (B, seq_len, feature_dim)
        """
        x = self.input_norm(self.x_embedder(x) + self.pos_embed)
        t_emb = self.t_embedder(t)
        img_emb = self.img_encoder(img)
        c = t_emb + img_emb
        
        for block in self.blocks:
            x = block(x, c)
        
        return self.final_layer(x)
