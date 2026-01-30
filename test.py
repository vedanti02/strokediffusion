"""
Test/Inference script for image-conditioned stroke generation.
Given an input image, generates strokes that reconstruct it.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

from src.config import (
    DEVICE, RENDERER_PATH,
    SEQ_LEN, FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    TOTAL_STROKES, DIFFUSION_STEPS, CANVAS_SIZE
)
from src.models import VisualStrokeDiT, NeuralRenderer
from src.models.renderer import load_renderer
from src.utils import sort_strokes_by_area


class ImageConditionedSampler:
    """Sampler for image-conditioned diffusion (VisualStrokeDiT)."""
    
    def __init__(self, num_steps: int = 1000, device: str = "cuda"):
        self.num_steps = num_steps
        self.device = device
        
        # Linear schedule (matches training)
        beta = torch.linspace(0.0001, 0.02, num_steps).to(device)
        self.alpha = 1. - beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # For posterior calculation
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_hat[:-1]])
        self.posterior_mean_coef1 = beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_mean_coef2 = (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_hat)
        self.posterior_variance = beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
    
    @torch.no_grad()
    def sample(
        self,
        model,
        target_image: torch.Tensor,
        cfg_scale: float = 1.0,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate strokes conditioned on target image.
        
        Args:
            model: Trained VisualStrokeDiT model
            target_image: Target image tensor (1, 3, 128, 128) in [0, 1]
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            verbose: Print progress
        
        Returns:
            Generated strokes (1, seq_len, feature_dim) in [-1, 1]
        """
        if verbose:
            print(f"Generating strokes with CFG scale {cfg_scale}...")
        
        model.eval()
        B = target_image.shape[0]
        
        # Start from noise
        x = torch.randn(B, SEQ_LEN, FEATURE_DIM).to(self.device)
        
        for i, t in enumerate(reversed(range(self.num_steps))):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            if cfg_scale > 1.0:
                # Classifier-free guidance
                # Conditional prediction
                pred_cond = model(x, t_batch, target_image)
                # Unconditional prediction (zero image)
                pred_uncond = model(x, t_batch, torch.zeros_like(target_image))
                # Guided prediction
                pred_x0 = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_x0 = model(x, t_batch, target_image)
            
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Posterior mean
            posterior_mean = (
                self.posterior_mean_coef1[t] * pred_x0 +
                self.posterior_mean_coef2[t] * x
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(self.posterior_variance[t]) * noise
            else:
                x = posterior_mean
            
            if verbose and i % 200 == 0:
                print(f"   Step {t}/{self.num_steps} | Range: [{x.min():.2f}, {x.max():.2f}]")
        
        return x


def load_image(image_path: str, size: int = 128) -> torch.Tensor:
    """Load and preprocess an image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img


def render_strokes_to_canvas(strokes: torch.Tensor, renderer, sort: bool = True) -> tuple:
    """
    Render strokes to canvas and return frames for GIF.
    
    Args:
        strokes: (N, 13) tensor in [0, 1]
        renderer: Neural renderer
        sort: Whether to sort by area
    
    Returns:
        Tuple of (final_canvas, list of PIL frames)
    """
    if sort:
        strokes = sort_strokes_by_area(strokes)
    
    canvas = torch.ones(1, 3, CANVAS_SIZE, CANVAS_SIZE).to(strokes.device)
    params, colors = strokes[:, :10], strokes[:, 10:]
    
    frames = []
    
    with torch.no_grad():
        alphas = renderer(params)  # (N, 1, H, W)
        
        # Initial white canvas
        img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
        
        for k in range(strokes.shape[0]):
            canvas = canvas * (1 - alphas[k]) + colors[k].view(3, 1, 1) * alphas[k]
            
            # Save frame every few strokes for GIF
            if k % 5 == 0 or k == strokes.shape[0] - 1:
                img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
                frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    return canvas, frames


def test(
    checkpoint_path: str,
    image_path: str,
    output_dir: str = "outputs",
    cfg_scale: float = 2.0,
    sort_strokes: bool = True
):
    """
    Main test function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to input image
        output_dir: Directory to save outputs
        cfg_scale: Classifier-free guidance scale
        sort_strokes: Whether to sort strokes by area before rendering
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- LOAD MODEL ---
    print("Loading model...")
    model = VisualStrokeDiT(
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"✓ Loaded: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return
    
    # --- LOAD RENDERER ---
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("✓ Renderer loaded")
    except Exception as e:
        print(f"⚠ Renderer error: {e}")
        return
    
    # --- LOAD INPUT IMAGE ---
    print(f"Loading image: {image_path}")
    target_image = load_image(image_path, CANVAS_SIZE).to(DEVICE)
    
    # --- GENERATE STROKES ---
    sampler = ImageConditionedSampler(num_steps=DIFFUSION_STEPS, device=DEVICE)
    generated_strokes = sampler.sample(model, target_image, cfg_scale=cfg_scale)
    
    # Normalize from [-1, 1] to [0, 1]
    strokes = torch.clamp((generated_strokes + 1) / 2, 0, 1)
    strokes = strokes.view(-1, 13)  # (100, 13)
    
    # --- RENDER ---
    print("Rendering strokes...")
    final_canvas, frames = render_strokes_to_canvas(strokes, renderer, sort=sort_strokes)
    
    # --- SAVE OUTPUTS ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save GIF
    gif_path = os.path.join(output_dir, f"{base_name}_painting.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"✓ Saved: {gif_path}")
    
    # Save final painting
    final_path = os.path.join(output_dir, f"{base_name}_final.png")
    frames[-1].save(final_path)
    print(f"✓ Saved: {final_path}")
    
    # Save strokes tensor
    strokes_path = os.path.join(output_dir, f"{base_name}_strokes.pt")
    torch.save(strokes.cpu(), strokes_path)
    print(f"✓ Saved: {strokes_path}")
    
    # --- DISPLAY COMPARISON ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    orig = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(orig)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Painting
    painting = final_canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(painting)
    axes[1].set_title(f"Generated Painting ({TOTAL_STROKES} strokes)")
    axes[1].axis("off")
    
    # Difference
    diff = np.abs(orig - painting)
    axes[2].imshow(diff)
    axes[2].set_title("Difference")
    axes[2].axis("off")
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {comparison_path}")
    plt.show()
    
    # Calculate L2 loss
    l2_loss = F.mse_loss(final_canvas.cpu(), target_image.cpu()).item()
    print(f"\nL2 Reconstruction Loss: {l2_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate strokes from an input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG guidance scale (1.0 = no guidance)")
    parser.add_argument("--no_sort", action="store_true", help="Disable stroke sorting by area")
    args = parser.parse_args()
    
    test(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        sort_strokes=not args.no_sort
    )
