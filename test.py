"""
Progressive Stroke Painting — Inference  (conditional + unconditional)

Usage:
    # Conditional: paint a specific face
    python test.py --checkpoint best_model.pth --img face.jpg

    # Unconditional: generate random face paintings from noise
    python test.py --checkpoint best_model.pth --n_samples 4 --cfg_scale 3.0

    # Batch conditional
    python test.py --checkpoint best_model.pth --img_dir ./faces/

Outputs per sample:
  - *_evolution.gif  : step-by-step painting animation
  - *_final.png      : final painting
  - *_strokes.pt     : stroke parameters (100, 13) in [0, 1]
"""
import torch
import numpy as np
from PIL import Image
import os
import glob
import argparse
import math
from torchvision import transforms

from src.config import (
    DEVICE, RENDERER_PATH,
    NUM_STROKES, STROKE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    DIFFUSION_STEPS, CANVAS_SIZE, CFG_SCALE,
)
from src.models import ImageConditionedStrokeDiT
from src.models.renderer import load_renderer

NUM_GROUPS = 20
STROKES_PER_GROUP = 5


# ==========================================
# DDPM SAMPLER  (with classifier-free guidance)
# ==========================================
class DDPMSampler:
    """DDPM sampler with optional CFG for unconditional generation."""

    def __init__(self, num_steps=1000, device="cuda"):
        self.num_steps = num_steps
        self.device = device
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.posterior_log_var = torch.log(torch.clip(
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), 1e-20
        ))

    @torch.no_grad()
    def sample_group(
        self, model, canvas, target, step_idx,
        num_inference_steps=50, cfg_scale=1.0,
    ):
        """Denoise one group of strokes with optional CFG.

        If cfg_scale > 1 and target is not all-zeros, applies:
            pred = uncond + cfg_scale * (cond - uncond)

        If target is all-zeros (unconditional mode, cfg_scale=1),
        no guidance is needed — just denoise unconditionally.
        """
        B = canvas.shape[0]
        x = torch.randn(B, STROKES_PER_GROUP, STROKE_DIM, device=self.device)
        step_t = torch.full((B,), step_idx, device=self.device, dtype=torch.long)

        use_cfg = cfg_scale > 1.0 and target.abs().sum() > 0

        step_ratio = max(1, self.num_steps // num_inference_steps)
        timesteps = list(range(0, self.num_steps, step_ratio))[::-1]

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)

            if use_cfg:
                # Conditional prediction
                pred_cond = model(x, t_batch, canvas, target, step_t)
                # Unconditional prediction (target = zeros)
                null_target = torch.zeros_like(target)
                pred_uncond = model(x, t_batch, canvas, null_target, step_t)
                # CFG combination
                pred_x0 = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_x0 = model(x, t_batch, canvas, target, step_t)

            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            if t_val > 0:
                posterior_mean = (
                    self.posterior_mean_coef1[t_val] * pred_x0
                    + self.posterior_mean_coef2[t_val] * x
                )
                noise = torch.randn_like(x)
                x = posterior_mean + torch.exp(0.5 * self.posterior_log_var[t_val]) * noise
            else:
                x = pred_x0
        return x


# ==========================================
# RENDERING
# ==========================================
def render_group_on_canvas(canvas, group_strokes, renderer, canvas_size=128):
    B, num = group_strokes.shape[:2]
    strokes = torch.clamp(group_strokes, 0, 1)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    colors = strokes[:, :, 10:].view(B, num, 3, 1, 1)
    alphas = renderer(shape_params).view(B, num, 1, canvas_size, canvas_size)
    result = canvas
    for i in range(num):
        result = result * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    return result


def load_target_image(path, canvas_size=128, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((canvas_size, canvas_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


# ==========================================
# CORE GENERATION
# ==========================================
def generate_painting(model, renderer, sampler, target, canvas_size, device,
                      num_inference_steps=50, cfg_scale=1.0):
    """Progressive stroke generation for one sample.

    Args:
        target: (1, 3, H, W) target image. All-zeros for unconditional.
        cfg_scale: guidance scale (>1 boosts conditional, 1 = no guidance)
    Returns:
        all_strokes: (1, 100, 13) in [0, 1]
        frames: list of PIL Images
    """
    all_strokes = torch.zeros(1, NUM_STROKES, STROKE_DIM, device=device)
    frames = []

    canvas = torch.ones(1, 3, canvas_size, canvas_size, device=device)
    img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frames.append(Image.fromarray((img * 255).astype(np.uint8)))

    for step in range(NUM_GROUPS):
        group = sampler.sample_group(
            model, canvas, target, step, num_inference_steps, cfg_scale
        )
        group_01 = torch.clamp((group + 1) / 2, 0, 1)

        start = step * STROKES_PER_GROUP
        all_strokes[0, start : start + STROKES_PER_GROUP] = group_01.squeeze(0)
        canvas = render_group_on_canvas(canvas, group_01, renderer, canvas_size)

        img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))

    return all_strokes, frames


def save_outputs(name, frames, all_strokes, output_dir, canvas_size, target_pil=None):
    """Save GIF, final PNG, comparison, and strokes .pt."""
    # Evolution GIF
    gif_frames = ([target_pil] + frames) if target_pil else frames
    gif_path = os.path.join(output_dir, f"{name}_evolution.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=200, loop=0)
    print(f"  GIF: {gif_path}")

    # Final painting
    final_path = os.path.join(output_dir, f"{name}_final.png")
    frames[-1].save(final_path)
    print(f"  Final: {final_path}")

    # Side-by-side comparison (only if conditional)
    if target_pil is not None:
        side = Image.new("RGB", (canvas_size * 2, canvas_size))
        side.paste(target_pil, (0, 0))
        side.paste(frames[-1], (canvas_size, 0))
        side.save(os.path.join(output_dir, f"{name}_comparison.png"))

    # Strokes tensor
    torch.save(all_strokes.squeeze(0).cpu(), os.path.join(output_dir, f"{name}_strokes.pt"))

    strokes = all_strokes.squeeze(0)
    print(f"  Strokes: {tuple(strokes.shape)} | "
          f"pos [{strokes[:,:6].min():.3f}, {strokes[:,:6].max():.3f}] | "
          f"color [{strokes[:,10:].min():.3f}, {strokes[:,10:].max():.3f}]")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Progressive stroke painting (conditional + unconditional)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img", type=str, default=None, help="Target face image (conditional)")
    parser.add_argument("--img_dir", type=str, default=None, help="Dir of target faces (conditional)")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of unconditional samples")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG scale (default from config)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inference_steps", type=int, default=50)
    args = parser.parse_args()

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else CFG_SCALE
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Determine mode
    is_conditional = args.img is not None or args.img_dir is not None

    # Load model
    print("Loading model...")
    model = ImageConditionedStrokeDiT(
        stroke_dim=STROKE_DIM, strokes_per_group=STROKES_PER_GROUP,
        num_groups=NUM_GROUPS, d_model=D_MODEL, nhead=NUM_HEADS, num_layers=NUM_LAYERS,
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    renderer = load_renderer(RENDERER_PATH, DEVICE)
    sampler = DDPMSampler(num_steps=DIFFUSION_STEPS, device=DEVICE)

    if is_conditional:
        # ---- CONDITIONAL: paint given face(s) ----
        image_paths = []
        if args.img:
            image_paths.append(args.img)
        if args.img_dir:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                image_paths.extend(sorted(glob.glob(os.path.join(args.img_dir, ext))))

        print(f"\nConditional mode: {len(image_paths)} image(s), cfg_scale={cfg_scale}")
        for idx, ip in enumerate(image_paths):
            name = os.path.splitext(os.path.basename(ip))[0]
            print(f"\n[{idx+1}/{len(image_paths)}] {name}")

            target = load_target_image(ip, CANVAS_SIZE, DEVICE)
            target_pil = Image.fromarray(
                (target.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            all_strokes, frames = generate_painting(
                model, renderer, sampler, target, CANVAS_SIZE, DEVICE,
                args.inference_steps, cfg_scale,
            )
            save_outputs(name, frames, all_strokes, args.output_dir, CANVAS_SIZE, target_pil)
    else:
        # ---- UNCONDITIONAL: generate random face paintings ----
        null_target = torch.zeros(1, 3, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
        print(f"\nUnconditional mode: {args.n_samples} sample(s), cfg_scale=1.0 (no guidance)")

        for idx in range(args.n_samples):
            name = f"uncond_{idx:04d}"
            if args.seed is not None:
                name = f"seed{args.seed}_{name}"
            print(f"\n[{idx+1}/{args.n_samples}] {name}")

            all_strokes, frames = generate_painting(
                model, renderer, sampler, null_target, CANVAS_SIZE, DEVICE,
                args.inference_steps, cfg_scale=1.0,  # no guidance for pure unconditional
            )
            save_outputs(name, frames, all_strokes, args.output_dir, CANVAS_SIZE)

    print(f"\nDone! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
