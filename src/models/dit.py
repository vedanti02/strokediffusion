"""
Progressive Visual Stroke DiT for painter-like stroke generation.
Generates strokes group by group, each conditioned on the current canvas.
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
    """Encode canvas images into conditioning vectors."""
    
    def __init__(self, output_dim: int = 384):
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
        return self.net(x)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (adaLN) modulation."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        
        # MLP with modulation
        return x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class StrokeEmbedding(nn.Module):
    """Embed strokes into transformer dimension with positional encoding."""
    
    def __init__(self, stroke_dim: int = 13, hidden_size: int = 384, num_strokes: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(stroke_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_strokes, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.pos_embed


class ProgressiveVisualStrokeDiT(nn.Module):
    """
    Progressive DiT for painter-like stroke generation.
    Generates one group of strokes at a time, conditioned on the current canvas.
    
    Input: (B, strokes_per_group, 13) noisy strokes + timestep + canvas image + group index
    Output: (B, strokes_per_group, 13) predicted clean strokes for this group
    """
    
    def __init__(
        self,
        stroke_dim: int = 13,
        strokes_per_group: int = 5,
        num_groups: int = 20,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.strokes_per_group = strokes_per_group
        self.num_groups = num_groups
        
        # Stroke embedding for group of strokes
        self.stroke_embed = StrokeEmbedding(stroke_dim, d_model, strokes_per_group)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(d_model)
        
        # Canvas encoder (sees painting so far)
        self.img_encoder = ImageEncoder(output_dim=d_model)
        
        # Group index embedding: tells model which group (0-19) we're generating
        self.group_embed = nn.Embedding(num_groups, d_model)
        
        self.input_norm = nn.LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Output projection back to stroke params
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, stroke_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        canvas: torch.Tensor, 
        group_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy strokes for this group (B, strokes_per_group, stroke_dim)
            t: Diffusion timestep (B,)
            canvas: Current canvas image from previous strokes (B, 3, 128, 128)
            group_idx: Which group we're generating, 0 to num_groups-1 (B,)
        Returns:
            Predicted clean strokes for this group (B, strokes_per_group, stroke_dim)
        """
        # Embed strokes
        x = self.input_norm(self.stroke_embed(x))
        
        # Conditioning: timestep + canvas + group index
        t_emb = self.t_embedder(t)
        canvas_emb = self.img_encoder(canvas)
        group_emb = self.group_embed(group_idx)
        
        c = t_emb + canvas_emb + group_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        return self.final_layer(x)
