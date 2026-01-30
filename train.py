"""
Progressive Stroke Diffusion Training with VisualStrokeDiT

Key Idea:
- Strokes are generated progressively in groups (20 groups x 5 strokes = 100 total)
- Each group is conditioned on the ACCUMULATED painting from previous strokes
- Uses ProgressiveVisualStrokeDiT (extends VisualStrokeDiT with group embedding)

Training:
- For each group k (out of 20):
  1. Render strokes from groups 0..k-1 → get canvas_k
  2. Add diffusion noise to group k's strokes
  3. Model predicts clean group k strokes given (noisy_strokes_k, t, canvas_k, group_idx=k)

Like a painter: paint → look at canvas → paint next meaningful stroke
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import os
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from src.config import (
    DEVICE, DATA_PATH, RENDERER_PATH,
    NUM_STROKES, STROKE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, CANVAS_SIZE,
    get_save_dir
)
from src.models import ProgressiveVisualStrokeDiT, PerceptualLoss
from src.models.renderer import load_renderer
from src.data import get_dataloader

# ==========================================
# CONFIGURATION
# ==========================================
NUM_GROUPS = 20           # 20 groups of 5 strokes = 100 total
STROKES_PER_GROUP = 5     # 5 strokes per group

USE_EMA = True
EMA_DECAY = 0.999
USE_AMP = True
GRADIENT_ACCUMULATION_STEPS = 2
PERCEPTUAL_WEIGHT = 0.25
STROKE_WEIGHT = 1.0


# ==========================================
# DIFFUSION SCHEDULER
# ==========================================
class DiffusionScheduler:
    def __init__(self, num_steps=1000, device="cuda"):
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        self.alphas_cumprod = torch.cumprod(1. - self.betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    def add_noise(self, original, t):
        """Add noise to original data at timestep t."""
        noise = torch.randn_like(original)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Handle different batch shapes
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        noisy = sqrt_alpha * original + sqrt_one_minus * noise
        return noisy, noise


# ==========================================
# RENDERING UTILITIES
# ==========================================
def render_strokes_progressive(all_strokes, renderer, up_to_group, canvas_size=128):
    """
    Render strokes from groups 0 to up_to_group-1.
    
    Args:
        all_strokes: (B, NUM_STROKES, 13) all strokes in [0, 1]
        renderer: Neural renderer
        up_to_group: Render groups 0..up_to_group-1
        
    Returns:
        canvas: (B, 3, H, W) rendered image
    """
    B = all_strokes.shape[0]
    canvas = torch.ones(B, 3, canvas_size, canvas_size, device=all_strokes.device)
    
    if up_to_group == 0:
        return canvas  # Blank canvas for first group
    
    # Render strokes from groups 0 to up_to_group-1
    num_strokes_to_render = up_to_group * STROKES_PER_GROUP
    strokes_to_render = all_strokes[:, :num_strokes_to_render, :]  # (B, N, 13)
    
    strokes_to_render = torch.clamp(strokes_to_render, 0, 1)
    shape_params = strokes_to_render[:, :, :10].reshape(-1, 10)
    color_params = strokes_to_render[:, :, 10:]
    
    alphas = renderer(shape_params).view(B, num_strokes_to_render, 1, canvas_size, canvas_size)
    colors = color_params.view(B, num_strokes_to_render, 3, 1, 1)
    
    for i in range(num_strokes_to_render):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    
    return canvas


def render_all_strokes(strokes, renderer, canvas_size=128, num_strokes=100):
    """Render all strokes at once."""
    B = strokes.shape[0]
    strokes = torch.clamp(strokes, 0, 1)
    
    strokes = strokes.view(B, num_strokes, 13)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    color_params = strokes[:, :, 10:]
    
    alphas = renderer(shape_params).view(B, num_strokes, 1, canvas_size, canvas_size)
    canvas = torch.ones(B, 3, canvas_size, canvas_size, device=strokes.device)
    colors = color_params.view(B, num_strokes, 3, 1, 1)
    
    for i in range(num_strokes):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    
    return canvas


# ==========================================
# TRAINING
# ==========================================
def train(
    data_path: str = None,
    batch_size: int = None,
    learning_rate: float = None,
    epochs: int = None,
    resume_from: str = None
):
    data_path = data_path or DATA_PATH
    batch_size = batch_size or BATCH_SIZE
    learning_rate = learning_rate or LEARNING_RATE
    epochs = epochs or EPOCHS
    
    save_dir = get_save_dir("progressive_dit")
    
    # --- LOAD RENDERER ---
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("[OK] Renderer loaded")
    except Exception as e:
        print(f"[WARN] Renderer not found: {e}")
        return
    
    # --- SETUP DATA ---
    print(f"Loading data from: {data_path}")
    dataloader = get_dataloader(
        data_path, 
        batch_size=batch_size,
        num_strokes=NUM_STROKES,
        stroke_dim=STROKE_DIM
    )
    
    # --- SETUP MODEL ---
    print(f"\nModel: ProgressiveVisualStrokeDiT")
    print(f"  - {NUM_GROUPS} groups x {STROKES_PER_GROUP} strokes = {NUM_STROKES} total")
    print(f"  - d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
    
    model = ProgressiveVisualStrokeDiT(
        stroke_dim=STROKE_DIM,
        strokes_per_group=STROKES_PER_GROUP,
        num_groups=NUM_GROUPS,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Parameters: {num_params:,}")
    
    # Resume
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"[OK] Resumed from: {resume_from}")
    
    # --- LOSSES ---
    perceptual_criterion = PerceptualLoss().to(DEVICE)
    
    # --- OPTIMIZER ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    
    # --- EMA ---
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)) if USE_EMA else None
    
    # --- AMP ---
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # --- DIFFUSION ---
    diff_sched = DiffusionScheduler(device=DEVICE)
    
    print(f"\nTraining config:")
    print(f"  - Data: {len(dataloader.dataset)} samples")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Output: {save_dir}")
    print()
    print("Training: For each batch, randomly pick group k,")
    print("render groups 0..k-1, then train diffusion on group k.")
    print()
    
    best_loss = float('inf')
    
    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch_idx, clean_strokes in enumerate(pbar):
            # clean_strokes: (B, NUM_STROKES, 13) in [-1, 1]
            clean_strokes = clean_strokes.to(DEVICE)
            B = clean_strokes.shape[0]
            
            # Convert to [0, 1] for rendering
            clean_strokes_01 = torch.clamp((clean_strokes + 1) / 2, 0, 1)
            
            # Randomly pick which group to train on for each sample
            group_idx = torch.randint(0, NUM_GROUPS, (B,), device=DEVICE)
            
            # Get the strokes for this group: (B, STROKES_PER_GROUP, 13)
            group_strokes_list = []
            for i in range(B):
                g = group_idx[i].item()
                start = g * STROKES_PER_GROUP
                end = start + STROKES_PER_GROUP
                group_strokes_list.append(clean_strokes[i, start:end, :])  # (5, 13)
            group_strokes = torch.stack(group_strokes_list)  # (B, 5, 13) in [-1, 1]
            
            # Diffusion timestep
            t = torch.randint(0, 1000, (B,), device=DEVICE).long()
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # Add noise to this group's strokes
                noisy_group, _ = diff_sched.add_noise(group_strokes, t)  # (B, 5, 13)
                
                # Render canvas from previous groups (groups 0..k-1)
                with torch.no_grad():
                    canvas_list = []
                    for i in range(B):
                        g = group_idx[i].item()
                        c = render_strokes_progressive(
                            clean_strokes_01[i:i+1], renderer,
                            up_to_group=g,
                            canvas_size=CANVAS_SIZE
                        )
                        canvas_list.append(c)
                    canvas = torch.cat(canvas_list, dim=0)  # (B, 3, 128, 128)
                
                # Model predicts clean strokes for this group
                predicted_group = model(noisy_group, t, canvas, group_idx)  # (B, 5, 13)
                
                # Stroke MSE loss
                stroke_loss = F.mse_loss(predicted_group, group_strokes)
                
                # Perceptual loss: render predicted vs target (full painting)
                full_predicted = clean_strokes_01.clone()
                for i in range(B):
                    g = group_idx[i].item()
                    start = g * STROKES_PER_GROUP
                    pred_01 = torch.clamp((predicted_group[i] + 1) / 2, 0, 1)
                    full_predicted[i, start:start+STROKES_PER_GROUP, :] = pred_01
                
                pred_rendered = render_all_strokes(full_predicted, renderer, CANVAS_SIZE, NUM_STROKES)
                target_rendered = render_all_strokes(clean_strokes_01, renderer, CANVAS_SIZE, NUM_STROKES)
                perceptual_loss = perceptual_criterion(pred_rendered, target_rendered)
                
                # Total loss
                total_loss = STROKE_WEIGHT * stroke_loss + PERCEPTUAL_WEIGHT * perceptual_loss
                total_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
            
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
            pbar.set_postfix({'loss': f"{stroke_loss.item():.4f}"})
        
        scheduler_lr.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.5f} | LR: {scheduler_lr.get_last_lr()[0]:.2e}")
        
        # Save
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"   [OK] Best model saved! Loss: {best_loss:.6f}")
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            }, os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
    
    # Save EMA model
    if USE_EMA:
        torch.save({
            'model_state_dict': ema_model.module.state_dict(),
            'epoch': epochs
        }, os.path.join(save_dir, "ema_model.pth"))
        print(f"[OK] EMA model saved")
    
    print(f"\n[OK] Training Complete! Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    train(args.data_path, args.batch_size, args.lr, args.epochs, args.resume)
