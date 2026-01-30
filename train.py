"""
Training script for image-conditioned stroke diffusion.
Uses VisualStrokeDiT with perceptual loss.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

from src.config import (
    DEVICE, DATA_PATH, RENDERER_PATH,
    SEQ_LEN, FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    PERCEPTUAL_WEIGHT, CFG_DROP_PROB,
    get_save_dir
)
from src.models import VisualStrokeDiT, NeuralRenderer, PerceptualLoss
from src.models.renderer import load_renderer
from src.data import get_dataloader
from src.diffusion import Diffusion
from src.utils import render_batch


def train():
    save_dir = get_save_dir("face_reconstruction")
    
    # --- LOAD RENDERER ---
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("✓ Renderer loaded")
    except Exception as e:
        print(f"⚠ Renderer not found: {e}")
        return
    
    # --- SETUP ---
    dataloader = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)
    
    model = VisualStrokeDiT(
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    perceptual_criterion = PerceptualLoss().to(DEVICE)
    diffusion = Diffusion(device=DEVICE)
    
    print(f"Training on {len(dataloader.dataset)} files. Output saved to {save_dir}")
    
    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for clean_strokes in pbar:
            clean_strokes = clean_strokes.to(DEVICE)
            B = clean_strokes.shape[0]
            t = torch.randint(0, 1000, (B,), device=DEVICE).long()
            
            # 1. Generate target image from clean strokes
            with torch.no_grad():
                target_image = render_batch(clean_strokes, renderer)
            
            # 2. Add noise to strokes
            noisy_strokes, noise = diffusion.noise(clean_strokes, t)
            
            # 3. Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                # CFG: Randomly drop image condition
                if torch.rand(1) < CFG_DROP_PROB:
                    cond_input = torch.zeros_like(target_image)
                else:
                    cond_input = target_image
                
                predicted_strokes = model(noisy_strokes, t, cond_input)
                
                # 4. Compute loss
                loss_mse = F.mse_loss(predicted_strokes, clean_strokes)
                pred_image = render_batch(predicted_strokes, renderer)
                loss_vgg = perceptual_criterion(pred_image, target_image)
                loss = loss_mse + (PERCEPTUAL_WEIGHT * loss_vgg)
            
            # 5. Optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # --- SAVE & DEBUG ---
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}. Loss: {avg_loss:.5f}")
        
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch}.pth")
            vutils.save_image(target_image, f"{save_dir}/debug_target_epoch_{epoch}.png")
            vutils.save_image(pred_image, f"{save_dir}/debug_pred_epoch_{epoch}.png")
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/model_final.pth")
    print("Training complete!")


if __name__ == "__main__":
    train()
