"""
Progressive Stroke DiT with Classifier-Free Guidance.

Trains on (face image, strokes) pairs.  At inference, supports:
  - Conditional:   paint a specific target face
  - Unconditional: generate a novel face painting from noise (via CFG)
"""
import torch
import torch.nn as nn
import math


# =====================================================================
# Shared building blocks
# =====================================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep encoding → MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
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


class StrokeEmbedding(nn.Module):
    """Project raw stroke params into transformer dim + learned positional encoding."""

    def __init__(self, stroke_dim: int = 13, hidden_size: int = 384, num_strokes: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(stroke_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_strokes, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.pos_embed


# =====================================================================
# Spatial Image Encoder
# =====================================================================
class SpatialImageEncoder(nn.Module):
    """Encode [canvas, target, target−canvas] → 64 spatial tokens (8×8).

    When target is all zeros (unconditional / CFG-dropped), the delta
    channel becomes −canvas — the model learns to interpret this as
    'no external reference, imagine on your own.'

    Input:  canvas (B,3,128,128) + target (B,3,128,128)
    Output: (B, 64, d_model)
    """

    def __init__(self, d_model: int = 384):
        super().__init__()
        # 9 channels = canvas(3) + target(3) + delta(3)
        self.convs = nn.Sequential(
            nn.Conv2d(9, 64, 4, stride=2, padding=1),  nn.SiLU(),   # 128→64
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),  # 64→32
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(), # 32→16
            nn.Conv2d(256, d_model, 4, stride=2, padding=1), nn.SiLU(),  # 16→8
        )
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = target - canvas                                # what still needs painting
        x = torch.cat([canvas, target, delta], dim=1)          # (B, 9, 128, 128)
        features = self.convs(x)                               # (B, d_model, 8, 8)
        B, C, H, W = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 64, d_model)
        return self.norm(tokens + self.pos_embed)


# =====================================================================
# DiT Block with Cross-Attention
# =====================================================================
class DiTCrossAttnBlock(nn.Module):
    """Transformer block: self-attn (adaLN) → cross-attn (spatial) → MLP (adaLN).

    • Self-attention with adaLN modulation (scalar conditioning: timestep + step)
    • Cross-attention where stroke queries attend to spatial image tokens
    • Feed-forward MLP with adaLN modulation
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # --- Self-attention (modulated by scalar conditioning) ---
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        # --- Cross-attention to spatial image features ---
        self.norm_cross_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_cross_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)

        # --- MLP (modulated by scalar conditioning) ---
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN: 4 modulation vectors for self-attn (shift, scale) + MLP (shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        c_scalar: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, N, D)  stroke tokens  (N = strokes_per_group, e.g. 5)
            c_scalar:   (B, D)     fused scalar conditioning (timestep + step)
            c_spatial:  (B, M, D)  spatial image tokens (M = 64)
        """
        shift_sa, scale_sa, shift_mlp, scale_mlp = self.adaLN_modulation(c_scalar).chunk(4, dim=1)

        # 1. Self-attention with adaLN modulation
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # 2. Cross-attention: strokes attend to spatial image features
        q = self.norm_cross_q(x)
        kv = self.norm_cross_kv(c_spatial)
        x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]

        # 3. MLP with adaLN modulation
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# =====================================================================
# Image-Conditioned Progressive Stroke DiT  (with CFG)
# =====================================================================
class ImageConditionedStrokeDiT(nn.Module):
    """
    Progressive stroke generation via diffusion + classifier-free guidance.

        strokes = f(noisy_strokes, t_diff, C_{t-1}, I_target, step)

    Training:
      - Sees (face image, strokes) pairs and learns face structure
      - Randomly drops target image with prob `cfg_dropout` (zeros it out)
        so the model also learns the unconditional distribution p(strokes | canvas, step)

    Inference:
      - Conditional:   give a face photo → paint it
      - Unconditional: target = zeros → the model 'imagines' a face
        using CFG:  pred = uncond + cfg_scale * (cond - uncond)
        The canvas (its own output) provides autoregressive memory.
    """

    def __init__(
        self,
        stroke_dim: int = 13,
        strokes_per_group: int = 5,
        num_groups: int = 20,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        cfg_dropout: float = 0.15,
    ):
        super().__init__()
        self.stroke_dim = stroke_dim
        self.strokes_per_group = strokes_per_group
        self.num_groups = num_groups
        self.cfg_dropout = cfg_dropout

        # Stroke input (noisy strokes being denoised)
        self.stroke_embed = StrokeEmbedding(stroke_dim, d_model, strokes_per_group)
        self.input_norm = nn.LayerNorm(d_model)

        # Scalar conditioning (adaLN): diffusion timestep + progressive step
        self.t_embedder = TimestepEmbedder(d_model)
        self.step_embed = nn.Embedding(num_groups, d_model)
        self.scalar_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Spatial conditioning (cross-attention): canvas + target + delta
        self.spatial_encoder = SpatialImageEncoder(d_model=d_model)

        # Transformer: self-attn (adaLN) + cross-attn (spatial) + MLP
        self.blocks = nn.ModuleList(
            [DiTCrossAttnBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)]
        )

        # Output → predicted clean stroke params
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, stroke_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        canvas: torch.Tensor,
        target: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:      Noisy strokes  (B, strokes_per_group, stroke_dim)
            t:      Diffusion timestep  (B,)
            canvas: Current canvas C_{t-1}  (B, 3, H, W)  — always real
            target: Target face image  (B, 3, H, W)  — zeroed for unconditional
            step:   Progressive step index  (B,), values in [0, num_groups-1]
        Returns:
            Predicted clean strokes  (B, strokes_per_group, stroke_dim)
        """
        # CFG dropout: randomly zero-out target during training
        if self.training and self.cfg_dropout > 0:
            B = target.shape[0]
            drop_mask = (torch.rand(B, device=target.device) < self.cfg_dropout).float()
            target = target * (1 - drop_mask[:, None, None, None])

        # Embed noisy strokes
        x = self.input_norm(self.stroke_embed(x))

        # Scalar conditioning for adaLN
        c_scalar = self.scalar_fuse(
            torch.cat([self.t_embedder(t), self.step_embed(step)], dim=-1)
        )

        # Spatial conditioning for cross-attention
        c_spatial = self.spatial_encoder(canvas, target)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c_scalar, c_spatial)

        return self.final_layer(x)
