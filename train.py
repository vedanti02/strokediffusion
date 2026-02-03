import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import sys
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.utils import save_image

# Assume these exist in your project structure
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
NUM_GROUPS = 20           
STROKES_PER_GROUP = 5     

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
        noise = torch.randn_like(original)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        noisy = sqrt_alpha * original + sqrt_one_minus * noise
        return noisy, noise

# ==========================================
# RENDERING UTILITIES
# ==========================================
def render_strokes_progressive(all_strokes, renderer, up_to_group, canvas_size=128):
    B = all_strokes.shape[0]
    canvas = torch.ones(B, 3, canvas_size, canvas_size, device=all_strokes.device)
    if up_to_group == 0: return canvas
    
    num_strokes_to_render = up_to_group * STROKES_PER_GROUP
    strokes_to_render = all_strokes[:, :num_strokes_to_render, :]
    
    strokes_to_render = torch.clamp(strokes_to_render, 0, 1)
    shape_params = strokes_to_render[:, :, :10].reshape(-1, 10)
    color_params = strokes_to_render[:, :, 10:]
    
    alphas = renderer(shape_params).view(B, num_strokes_to_render, 1, canvas_size, canvas_size)
    colors = color_params.view(B, num_strokes_to_render, 3, 1, 1)
    
    for i in range(num_strokes_to_render):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    return canvas

def render_all_strokes(strokes, renderer, canvas_size=128, num_strokes=100):
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
# VISUALIZATION UTILS
# ==========================================
def _save_epoch_visual(epoch, model, renderer, dataloader, save_dir, device):
    """
    Saves a visualization: [Context | Prediction | Ground Truth]
    """
    model.eval()
    with torch.no_grad():
        try:
            # Get a small batch (8 images max)
            clean_strokes = next(iter(dataloader))
            clean_strokes = clean_strokes.to(device)[:8] 
        except Exception:
            return

        B = clean_strokes.shape[0]
        
        # Visualize Group 10 (Middle of process)
        group_idx_val = 10 
        group_idx = torch.full((B,), group_idx_val, device=device, dtype=torch.long)
        
        # Context
        clean_01 = torch.clamp((clean_strokes + 1) / 2, 0, 1)
        canvas = render_strokes_progressive(clean_01, renderer, group_idx_val, CANVAS_SIZE)
        
        # Target Group
        start = group_idx_val * STROKES_PER_GROUP
        end = start + STROKES_PER_GROUP
        target_group = clean_strokes[:, start:end, :]
        
        # Noisy Input
        noise = torch.randn_like(target_group)
        t = torch.full((B,), 500, device=device, dtype=torch.long)
        noisy_group = 0.7 * target_group + 0.7 * noise 
        
        # Predict
        pred_strokes = model(noisy_group, t, canvas, group_idx)
        
        # Render Prediction
        full_pred_strokes = clean_01.clone()
        pred_01 = torch.clamp((pred_strokes + 1) / 2, 0, 1)
        full_pred_strokes[:, start:end, :] = pred_01
        
        pred_img = render_all_strokes(full_pred_strokes, renderer, CANVAS_SIZE, NUM_STROKES)
        gt_img = render_all_strokes(clean_01, renderer, CANVAS_SIZE, NUM_STROKES)
        
        # Stack & Save
        combined = torch.cat([canvas, pred_img, gt_img], dim=3)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_visual.jpg")
        save_image(combined, save_path, nrow=2, normalize=True)

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
    
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("[OK] Renderer loaded")
    except Exception as e:
        print(f"[WARN] Renderer not found: {e}")
        return
    
    print(f"Loading data from: {data_path}")
    dataloader = get_dataloader(
        data_path, 
        batch_size=batch_size,
        num_strokes=NUM_STROKES,
        stroke_dim=STROKE_DIM
    )
    
    print(f"\nModel: ProgressiveVisualStrokeDiT")
    model = ProgressiveVisualStrokeDiT(
        stroke_dim=STROKE_DIM,
        strokes_per_group=STROKES_PER_GROUP,
        num_groups=NUM_GROUPS,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"[OK] Resumed from: {resume_from}")
    
    perceptual_criterion = PerceptualLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)) if USE_EMA else None
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    diff_sched = DiffusionScheduler(device=DEVICE)
    
    print(f"Training started on {DEVICE}...")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_stroke_loss = 0.0
        epoch_perceptual_loss = 0.0
        optimizer.zero_grad()
        
        # Iterate silently (NO tqdm)
        for batch_idx, clean_strokes in enumerate(dataloader):
            clean_strokes = clean_strokes.to(DEVICE)
            B = clean_strokes.shape[0]
            clean_strokes_01 = torch.clamp((clean_strokes + 1) / 2, 0, 1)
            
            group_idx = torch.randint(0, NUM_GROUPS, (B,), device=DEVICE)
            
            group_strokes_list = []
            for i in range(B):
                g = group_idx[i].item()
                start = g * STROKES_PER_GROUP
                end = start + STROKES_PER_GROUP
                group_strokes_list.append(clean_strokes[i, start:end, :])
            group_strokes = torch.stack(group_strokes_list)
            
            t = torch.randint(0, 1000, (B,), device=DEVICE).long()
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                noisy_group, _ = diff_sched.add_noise(group_strokes, t)
                
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
                    canvas = torch.cat(canvas_list, dim=0)
                
                predicted_group = model(noisy_group, t, canvas, group_idx)
                
                stroke_loss = F.mse_loss(predicted_group, group_strokes)
                
                # Perceptual Loss Calculation
                full_predicted = clean_strokes_01.clone()
                for i in range(B):
                    g = group_idx[i].item()
                    start = g * STROKES_PER_GROUP
                    pred_01 = torch.clamp((predicted_group[i] + 1) / 2, 0, 1)
                    full_predicted[i, start:start+STROKES_PER_GROUP, :] = pred_01
                
                pred_rendered = render_all_strokes(full_predicted, renderer, CANVAS_SIZE, NUM_STROKES)
                target_rendered = render_all_strokes(clean_strokes_01, renderer, CANVAS_SIZE, NUM_STROKES)
                perceptual_loss = perceptual_criterion(pred_rendered, target_rendered)
                
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

            # Accumulate (Need to multiply back by steps because loss is averaged)
            epoch_loss += total_loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_stroke_loss += stroke_loss.item() 
            epoch_perceptual_loss += perceptual_loss.item() 
        
        # End of Epoch
        scheduler_lr.step()
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_stroke = epoch_stroke_loss / num_batches
        avg_percept = epoch_perceptual_loss / num_batches

        # --- ONE LOG LINE PER EPOCH ---
        print(
            f"Epoch {epoch+1}/{epochs} | Total: {avg_loss:.5f} | Stroke: {avg_stroke:.5f} | Percept: {avg_percept:.5f} | LR: {scheduler_lr.get_last_lr()[0]:.2e}",
            file=sys.stdout,
            flush=True
        )

        # --- SAVE VISUALIZATION EVERY 20 EPOCHS ---
        if (epoch + 1) % 20 == 0:
            model_to_use = ema_model.module if (USE_EMA and hasattr(ema_model, 'module')) else (ema_model if USE_EMA else model)
            _save_epoch_visual(epoch, model_to_use, renderer, dataloader, save_dir, DEVICE)

        # --- SAVE CHECKPOINT IF BEST ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, os.path.join(save_dir, "best_model.pth"))
            # Optional: print helper to confirm save without spamming
            # print(f"   [Saved Best Model]", flush=True)

    if USE_EMA:
        torch.save({
            'model_state_dict': ema_model.module.state_dict(),
            'epoch': epochs
        }, os.path.join(save_dir, "ema_model.pth"))
    
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