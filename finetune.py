"""
Fine-tuning script for polishing a trained model with MSE loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from src.config import (
    DEVICE, DATA_PATH,
    SEQ_LEN, FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    BATCH_SIZE, get_save_dir
)
from src.models import HeavyStrokeDiT
from src.data import get_dataloader
from src.diffusion import CosineScheduler


def finetune(
    checkpoint_path: str,
    epochs: int = 100,
    learning_rate: float = 1e-5
):
    save_dir = get_save_dir("finetuned")
    
    # --- SETUP ---
    dataloader = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)
    
    model = HeavyStrokeDiT(
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    # Load existing weights
    if os.path.exists(checkpoint_path):
        print(f"✓ Loaded weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("Train a model first!")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = CosineScheduler(device=DEVICE)
    
    print(f"Fine-tuning with MSE loss for {epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch}/{epochs}", leave=False)
        
        for clean_strokes in pbar:
            clean_strokes = clean_strokes.to(DEVICE)
            t = torch.randint(0, 1000, (clean_strokes.shape[0],), device=DEVICE).long()
            
            noisy_strokes, _ = scheduler.add_noise(clean_strokes, t)
            predicted = model(noisy_strokes, t)
            loss = criterion(predicted, clean_strokes)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} | MSE Loss: {avg_loss:.6f}")
        
        # Save
        torch.save(model.state_dict(), f"{save_dir}/model_polished.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{save_dir}/model_best.pth")
            print(f"   New best! ({best_loss:.6f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    
    finetune(args.checkpoint, args.epochs, args.lr)
