"""
Progressive Stroke Diffusion Training  (with Classifier-Free Guidance)

    strokes_t = f(noisy_strokes, t_diff, C_{t-1}, I_target, step)

Trains on (face image, stroke) pairs.  During training, the target image
is randomly dropped (zeroed) with probability `cfg_dropout` so the model
also learns the unconditional distribution p(strokes | canvas, step).

At inference:
  - Conditional:   paint a specific face photo
  - Unconditional: generate a novel face from pure noise (via CFG)
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import sys
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.utils import save_image

from src.config import (
    DEVICE, DATA_PATH, IMAGE_PATH, RENDERER_PATH,
    NUM_STROKES, STROKE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, CANVAS_SIZE, DIFFUSION_STEPS,
    CFG_DROPOUT,
    get_save_dir
)
from src.models import ImageConditionedStrokeDiT, PerceptualLoss
from src.models.renderer import load_renderer
from src.data import get_paired_dataloader

# ==========================================
# CONFIGURATION
# ==========================================
NUM_GROUPS = 20
STROKES_PER_GROUP = 5

USE_EMA = True
EMA_DECAY = 0.999
USE_AMP = True
GRADIENT_ACCUMULATION_STEPS = 2

# Loss weights
STROKE_WEIGHT = 1.0       # MSE on predicted clean strokes vs GT strokes
PIXEL_WEIGHT = 2.0        # L1 on rendered canvas delta (local)
PERCEPTUAL_WEIGHT = 0.1   # VGG perceptual on rendered canvas delta (local)

PATIENCE = 50


# ==========================================
# DIFFUSION SCHEDULER (cosine schedule)
# ==========================================
class DiffusionScheduler:
    """Cosine noise schedule for DDPM."""

    def __init__(self, num_steps=1000, device="cuda"):
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original, t):
        """q(x_t | x_0): add noise at timestep t."""
        noise = torch.randn_like(original)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * original + sqrt_one_minus * noise, noise


# ==========================================
# RENDERING UTILITIES
# ==========================================
def render_group_on_canvas(canvas, group_strokes, renderer, canvas_size=128):
    """Render a group of strokes ON TOP of an existing canvas.

    Args:
        canvas:        (B, 3, H, W) current canvas
        group_strokes: (B, strokes_per_group, 13) strokes in [0, 1]
        renderer:      pretrained neural stroke renderer
    Returns:
        Updated canvas (B, 3, H, W)
    """
    B, num = group_strokes.shape[:2]
    strokes = torch.clamp(group_strokes, 0, 1)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    colors = strokes[:, :, 10:].view(B, num, 3, 1, 1)
    alphas = renderer(shape_params).view(B, num, 1, canvas_size, canvas_size)
    result = canvas
    for i in range(num):
        result = result * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    return result


def render_gt_canvas(strokes_01, renderer, steps, canvas_size=128):
    """Render GT canvas up to each sample's progressive step.

    Args:
        strokes_01: (B, 100, 13) all GT strokes in [0, 1]
        renderer:   neural stroke renderer
        steps:      (B,) step per sample — render groups 0..step-1
    Returns:
        canvas (B, 3, H, W)
    """
    B = strokes_01.shape[0]
    canvases = []
    for i in range(B):
        s = steps[i].item()
        c = torch.ones(1, 3, canvas_size, canvas_size, device=strokes_01.device)
        if s > 0:
            c = render_group_on_canvas(
                c, strokes_01[i : i + 1, : s * STROKES_PER_GROUP, :], renderer, canvas_size
            )
        canvases.append(c)
    return torch.cat(canvases, dim=0)


def render_all_strokes(strokes, renderer, canvas_size=128, num_strokes=100):
    """Render all strokes from a blank canvas (for visualization)."""
    B = strokes.shape[0]
    strokes = torch.clamp(strokes, 0, 1).view(B, num_strokes, 13)
    canvas = torch.ones(B, 3, canvas_size, canvas_size, device=strokes.device)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    colors = strokes[:, :, 10:].view(B, num_strokes, 3, 1, 1)
    alphas = renderer(shape_params).view(B, num_strokes, 1, canvas_size, canvas_size)
    for i in range(num_strokes):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    return canvas


# ==========================================
# DDPM SAMPLER (for visualization rollout)
# ==========================================
class DDPMSampler:
    """Lightweight DDPM sampler for generating one group via denoising."""

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
    def sample_group(self, model, canvas, target, step_idx, num_inference_steps=50):
        """Generate one group of strokes via DDPM denoising.

        Args:
            model:   ImageConditionedStrokeDiT
            canvas:  (B, 3, H, W) current canvas
            target:  (B, 3, H, W) target face image
            step_idx: int, progressive step (0-19)
            num_inference_steps: denoising sub-steps (fewer = faster)
        Returns:
            (B, strokes_per_group, 13) predicted strokes in [-1, 1]
        """
        B = canvas.shape[0]
        x = torch.randn(B, STROKES_PER_GROUP, STROKE_DIM, device=self.device)
        step_t = torch.full((B,), step_idx, device=self.device, dtype=torch.long)

        step_ratio = max(1, self.num_steps // num_inference_steps)
        timesteps = list(range(0, self.num_steps, step_ratio))[::-1]

        for t_val in timesteps:
            t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            pred_x0 = model(x, t_batch, canvas, target, step_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            if t_val > 0:
                posterior_mean = self.posterior_mean_coef1[t_val] * pred_x0 + self.posterior_mean_coef2[t_val] * x
                noise = torch.randn_like(x)
                x = posterior_mean + torch.exp(0.5 * self.posterior_log_var[t_val]) * noise
            else:
                x = pred_x0
        return x


# ==========================================
# VISUALIZATION
# ==========================================
def _save_epoch_visual(epoch, model, renderer, sampler, dataloader, save_dir, device):
    """Progressive rollout: target | step5 | step10 | step15 | full_cond | uncond | GT."""
    model.eval()
    with torch.no_grad():
        try:
            clean_strokes, target_imgs = next(iter(dataloader))
            clean_strokes = clean_strokes.to(device)[:8]
            target_imgs = target_imgs.to(device)[:8]
        except Exception:
            return

        B = clean_strokes.shape[0]
        clean_01 = torch.clamp((clean_strokes + 1) / 2, 0, 1)
        if clean_01.shape[1] != NUM_STROKES:
            clean_01 = clean_01.view(B, NUM_STROKES, STROKE_DIM)

        gt_img = render_all_strokes(clean_01, renderer, CANVAS_SIZE, NUM_STROKES)

        # --- Conditional rollout ---
        vis_steps = [5, 10, 15, NUM_GROUPS]
        canvases = []
        canvas = torch.ones(B, 3, CANVAS_SIZE, CANVAS_SIZE, device=device)
        next_vis = 0

        for s in range(NUM_GROUPS):
            pred = sampler.sample_group(model, canvas, target_imgs, s, num_inference_steps=50)
            pred_01 = torch.clamp((pred + 1) / 2, 0, 1)
            canvas = render_group_on_canvas(canvas, pred_01, renderer, CANVAS_SIZE)
            if next_vis < len(vis_steps) and (s + 1) == vis_steps[next_vis]:
                canvases.append(canvas.clone())
                next_vis += 1

        # --- Unconditional rollout (target = zeros) ---
        null_target = torch.zeros_like(target_imgs)
        uncond_canvas = torch.ones(B, 3, CANVAS_SIZE, CANVAS_SIZE, device=device)
        for s in range(NUM_GROUPS):
            pred = sampler.sample_group(model, uncond_canvas, null_target, s, num_inference_steps=50)
            pred_01 = torch.clamp((pred + 1) / 2, 0, 1)
            uncond_canvas = render_group_on_canvas(uncond_canvas, pred_01, renderer, CANVAS_SIZE)

        # [target | step5 | step10 | step15 | full_cond | uncond | GT]
        combined = torch.cat([target_imgs] + canvases + [uncond_canvas, gt_img], dim=3)
        save_image(combined, os.path.join(save_dir, f"epoch_{epoch+1}_visual.jpg"), nrow=2, normalize=True)


# ==========================================
# TRAINING
# ==========================================
def train(
    data_path: str = None,
    image_path: str = None,
    batch_size: int = None,
    learning_rate: float = None,
    epochs: int = None,
    resume_from: str = None,
):
    data_path = data_path or DATA_PATH
    image_path = image_path or IMAGE_PATH
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE
    epochs = epochs or EPOCHS

    save_dir = get_save_dir("img_cond_progressive")

    print("="  * 70)
    print("PROGRESSIVE STROKE DIFFUSION  (CFG-enabled)")
    print("  strokes_t = f(noisy, t_diff, C_{t-1}, I_target, step)")
    print(f"  Groups: {NUM_GROUPS}  |  Strokes/group: {STROKES_PER_GROUP}")
    print(f"  Diffusion steps: {DIFFUSION_STEPS}  |  CFG dropout: {CFG_DROPOUT}")
    print(f"  Losses: stroke={STROKE_WEIGHT}  pixel={PIXEL_WEIGHT}  percept={PERCEPTUAL_WEIGHT}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    renderer = load_renderer(RENDERER_PATH, DEVICE)

    dataloader = get_paired_dataloader(
        stroke_dir=data_path,
        image_dir=image_path,
        batch_size=batch_size,
        num_strokes=NUM_STROKES,
        stroke_dim=STROKE_DIM,
        canvas_size=CANVAS_SIZE,
    )
    print(f"[OK] Paired dataset: {len(dataloader.dataset)} (stroke, image) pairs")

    model = ImageConditionedStrokeDiT(
        stroke_dim=STROKE_DIM,
        strokes_per_group=STROKES_PER_GROUP,
        num_groups=NUM_GROUPS,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        cfg_dropout=CFG_DROPOUT,
    ).to(DEVICE)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=DEVICE)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            model.load_state_dict(ckpt)
        print(f"[OK] Resumed from {resume_from} (epoch {start_epoch})")

    perceptual_criterion = PerceptualLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)) if USE_EMA else None
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    diff_sched = DiffusionScheduler(num_steps=DIFFUSION_STEPS, device=DEVICE)
    sampler = DDPMSampler(num_steps=DIFFUSION_STEPS, device=DEVICE)

    best_loss = float("inf")
    patience_counter = 0

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count / 1e6:.1f}M")
    print(f"Training on {DEVICE} — {len(dataloader)} batches/epoch\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_stroke = 0.0
        epoch_pixel = 0.0
        epoch_percept = 0.0
        optimizer.zero_grad()

        for batch_idx, (clean_strokes, target_imgs) in enumerate(dataloader):
            clean_strokes = clean_strokes.to(DEVICE)
            target_imgs = target_imgs.to(DEVICE)
            B = clean_strokes.shape[0]

            # Map strokes to [0, 1], ensure (B, 100, 13)
            clean_01 = torch.clamp((clean_strokes + 1) / 2, 0, 1)
            if clean_01.shape[1] != NUM_STROKES:
                clean_01 = clean_01.view(B, NUM_STROKES, STROKE_DIM)

            # ---- Sample random progressive step per sample ----
            step = torch.randint(0, NUM_GROUPS, (B,), device=DEVICE)

            # ---- Canvas from GT strokes up to this step ----
            with torch.no_grad():
                canvas = render_gt_canvas(clean_01, renderer, step, CANVAS_SIZE)

            # ---- GT strokes for this group (in [-1, 1]) ----
            gt_group_list = []
            for i in range(B):
                s = step[i].item()
                gt_group_list.append(
                    clean_strokes[i, s * STROKES_PER_GROUP : (s + 1) * STROKES_PER_GROUP]
                )
            gt_group = torch.stack(gt_group_list)  # (B, 5, 13)

            # ---- Diffusion: add noise to GT strokes ----
            t_diff = torch.randint(0, DIFFUSION_STEPS, (B,), device=DEVICE).long()
            noisy_group, _ = diff_sched.add_noise(gt_group, t_diff)

            # ---- Forward: denoise conditioned on (canvas, target, step) ----
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred_group = model(noisy_group, t_diff, canvas, target_imgs, step)

                # Loss 1: stroke parameter MSE (predicted clean vs actual clean)
                stroke_loss = F.mse_loss(pred_group, gt_group)

                # Loss 2 & 3: local rendered canvas losses
                pred_01 = torch.clamp((pred_group + 1) / 2, 0, 1)
                gt_01 = torch.clamp((gt_group + 1) / 2, 0, 1)

                canvas_after_pred = render_group_on_canvas(canvas, pred_01, renderer, CANVAS_SIZE)
                canvas_after_gt = render_group_on_canvas(canvas, gt_01, renderer, CANVAS_SIZE)

                pixel_loss = F.l1_loss(canvas_after_pred, canvas_after_gt)
                percept_loss = perceptual_criterion(canvas_after_pred, canvas_after_gt)

                total_loss = (
                    STROKE_WEIGHT * stroke_loss
                    + PIXEL_WEIGHT * pixel_loss
                    + PERCEPTUAL_WEIGHT * percept_loss
                ) / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if USE_EMA:
                    ema_model.update_parameters(model)

            epoch_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_stroke += stroke_loss.item()
            epoch_pixel += pixel_loss.item()
            epoch_percept += percept_loss.item()

        # -- End of epoch --
        scheduler_lr.step()
        N = max(len(dataloader), 1)
        avg_loss = epoch_loss / N
        avg_stroke = epoch_stroke / N
        avg_pixel = epoch_pixel / N
        avg_percept = epoch_percept / N

        print(
            f"Epoch {epoch+1}/{epochs} | Total: {avg_loss:.5f} "
            f"| Stroke: {avg_stroke:.5f} | Pixel: {avg_pixel:.5f} "
            f"| Percept: {avg_percept:.5f} "
            f"| LR: {scheduler_lr.get_last_lr()[0]:.2e}",
            flush=True,
        )

        # Visualize every 20 epochs
        if (epoch + 1) % 20 == 0:
            vis_model = ema_model.module if USE_EMA else model
            _save_epoch_visual(epoch, vis_model, renderer, sampler, dataloader, save_dir, DEVICE)

        # Checkpointing
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": best_loss},
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  Best! {best_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1}. Best: {best_loss:.6f}")
            break

    if USE_EMA:
        torch.save(
            {"model_state_dict": ema_model.module.state_dict(), "epoch": epochs},
            os.path.join(save_dir, "ema_model.pth"),
        )

    print(f"\n[OK] Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image-conditioned progressive stroke training")
    parser.add_argument("--data_path", type=str, default=None, help="Dir with stroke .pt files")
    parser.add_argument("--image_path", type=str, default=None, help="Dir with source face images")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args.data_path, args.image_path, args.batch_size, args.lr, args.epochs, args.resume)