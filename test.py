"""
Progressive Stroke Generation - Inference with VisualStrokeDiT

Generates strokes like a real painter:
1. Start with blank canvas
2. Generate group 1 (5 strokes) → render → see canvas
3. Generate group 2 conditioned on canvas → render → see canvas
4. ... repeat for all 20 groups
5. Final painting emerges stroke by stroke

Uses ProgressiveVisualStrokeDiT from src/models.
"""
import torch
import numpy as np
from PIL import Image
import os
import argparse
import math

from src.config import (
    DEVICE, RENDERER_PATH,
    NUM_STROKES, STROKE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    DIFFUSION_STEPS, CANVAS_SIZE
)
from src.models import ProgressiveVisualStrokeDiT
from src.models.renderer import load_renderer

# Must match training
NUM_GROUPS = 20
STROKES_PER_GROUP = 5


# ==========================================
# DIFFUSION SAMPLER
# ==========================================
class ProgressiveDiffusionSampler:
    """DDPM sampler for generating one group of strokes at a time."""
    
    def __init__(self, num_steps=1000, device="cuda"):
        self.num_steps = num_steps
        self.device = device
        
        # Cosine schedule
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_log_var = torch.log(torch.clip(
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), 1e-20, 1.0
        ))

    @torch.no_grad()
    def sample_group(self, model, canvas, group_idx, num_inference_steps=50):
        """
        Sample one group of strokes using DDPM.
        
        Args:
            model: ProgressiveVisualStrokeDiT
            canvas: (1, 3, 128, 128) current canvas image
            group_idx: int, which group (0-19)
            num_inference_steps: number of diffusion steps
        Returns:
            (1, STROKES_PER_GROUP, 13) generated strokes in [-1, 1]
        """
        # Start from noise: (1, 5, 13)
        x = torch.randn(1, STROKES_PER_GROUP, STROKE_DIM, device=self.device)
        g_idx = torch.tensor([group_idx], device=self.device)
        
        # Use fewer steps for speed (skip steps)
        step_ratio = max(1, self.num_steps // num_inference_steps)
        timesteps = list(range(0, self.num_steps, step_ratio))[::-1]
        
        for t in timesteps:
            t_batch = torch.tensor([t], device=self.device)
            
            # Predict clean strokes
            pred_x0 = model(x, t_batch, canvas, g_idx)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # DDPM update
            if t > 0:
                posterior_mean = (
                    self.posterior_mean_coef1[t] * pred_x0 +
                    self.posterior_mean_coef2[t] * x
                )
                noise = torch.randn_like(x)
                posterior_var = torch.exp(0.5 * self.posterior_log_var[t])
                x = posterior_mean + posterior_var * noise
            else:
                x = pred_x0
        
        return x  # (1, 5, 13)


# ==========================================
# RENDERING
# ==========================================
def render_strokes(all_strokes, renderer, canvas_size=128):
    """
    Render strokes to canvas.
    
    Args:
        all_strokes: (B, N, 13) strokes in [0, 1]
        renderer: Neural renderer
    Returns:
        canvas: (B, 3, H, W)
    """
    B = all_strokes.shape[0]
    num_strokes = all_strokes.shape[1]
    
    strokes = torch.clamp(all_strokes, 0, 1)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    color_params = strokes[:, :, 10:]
    
    alphas = renderer(shape_params).view(B, num_strokes, 1, canvas_size, canvas_size)
    canvas = torch.ones(B, 3, canvas_size, canvas_size, device=strokes.device)
    colors = color_params.view(B, num_strokes, 3, 1, 1)
    
    for i in range(num_strokes):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    
    return canvas


# ==========================================
# GENERATION
# ==========================================
def generate_progressive(
    checkpoint_path: str,
    output_dir: str = "outputs",
    n_samples: int = 1,
    seed: int = None,
    num_inference_steps: int = 50
):
    """
    Generate face paintings progressively, group by group.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    
    # Load model
    print("Loading ProgressiveVisualStrokeDiT model...")
    model = ProgressiveVisualStrokeDiT(
        stroke_dim=STROKE_DIM,
        strokes_per_group=STROKES_PER_GROUP,
        num_groups=NUM_GROUPS,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"[OK] Loaded checkpoint (epoch {epoch}): {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"[OK] Loaded: {checkpoint_path}")
    else:
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    model.eval()
    
    # Load renderer
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("[OK] Renderer loaded")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    sampler = ProgressiveDiffusionSampler(num_steps=1000, device=DEVICE)
    
    print(f"\nGenerating {n_samples} painting(s) progressively...")
    print(f"  - {NUM_GROUPS} groups x {STROKES_PER_GROUP} strokes = {NUM_STROKES} total")
    print(f"  - Inference steps per group: {num_inference_steps}")
    print()
    
    for sample_idx in range(n_samples):
        print(f"[Sample {sample_idx + 1}/{n_samples}]")
        
        # Storage for all strokes: (1, NUM_STROKES, 13) in [0, 1]
        all_strokes = torch.zeros(1, NUM_STROKES, STROKE_DIM, device=DEVICE)
        
        # Canvas evolution frames for GIF
        frames = []
        
        # Start with blank canvas
        canvas = torch.ones(1, 3, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
        img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
        
        # Generate group by group
        for group_idx in range(NUM_GROUPS):
            print(f"  Generating group {group_idx + 1}/{NUM_GROUPS}...", end="\r")
            
            # Sample this group conditioned on current canvas
            group_strokes = sampler.sample_group(
                model, canvas, group_idx, num_inference_steps
            )  # (1, 5, 13) in [-1, 1]
            
            # Convert to [0, 1]
            group_strokes_01 = torch.clamp((group_strokes + 1) / 2, 0, 1)
            
            # Store in all_strokes
            start = group_idx * STROKES_PER_GROUP
            all_strokes[0, start:start+STROKES_PER_GROUP, :] = group_strokes_01.squeeze(0)
            
            # Update canvas by rendering all strokes so far
            strokes_so_far = all_strokes[:, :start+STROKES_PER_GROUP, :]
            canvas = render_strokes(strokes_so_far, renderer, CANVAS_SIZE)
            
            # Save frame
            img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray((img * 255).astype(np.uint8)))
        
        print(f"  Generated all {NUM_GROUPS} groups!        ")
        
        # Save outputs
        sample_name = f"sample_{sample_idx:04d}"
        if seed is not None:
            sample_name = f"seed{seed}_{sample_name}"
        
        # GIF showing painting evolution
        gif_path = os.path.join(output_dir, f"{sample_name}_evolution.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"  [OK] Evolution GIF: {gif_path}")
        
        # Final painting
        final_path = os.path.join(output_dir, f"{sample_name}_final.png")
        frames[-1].save(final_path)
        print(f"  [OK] Final painting: {final_path}")
        
        # Strokes tensor
        strokes_path = os.path.join(output_dir, f"{sample_name}_strokes.pt")
        torch.save(all_strokes.squeeze(0).cpu(), strokes_path)
        print(f"  [OK] Strokes: {strokes_path}")
        
        # Stats
        strokes = all_strokes.squeeze(0)
        print(f"  Stroke stats:")
        print(f"    - Shape: {tuple(strokes.shape)}")
        print(f"    - Position range: [{strokes[:, :6].min():.3f}, {strokes[:, :6].max():.3f}]")
        print(f"    - Color range: [{strokes[:, 10:].min():.3f}, {strokes[:, 10:].max():.3f}]")
        print()
    
    print(f"[OK] Generation complete! {n_samples} painting(s) saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive stroke generation using VisualStrokeDiT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--inference_steps", type=int, default=50, help="Diffusion steps per group")
    args = parser.parse_args()
    
    generate_progressive(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        num_inference_steps=args.inference_steps
    )
