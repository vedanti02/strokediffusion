"""
Generate paintings from a trained model.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from src.config import (
    DEVICE, RENDERER_PATH,
    SEQ_LEN, FEATURE_DIM, D_MODEL, NUM_HEADS, NUM_LAYERS,
    TOTAL_STROKES, DIFFUSION_STEPS
)
from src.models import HeavyStrokeDiT, NeuralRenderer
from src.models.renderer import load_renderer
from src.diffusion import XStartScheduler
from src.utils import sort_strokes_by_area


def generate(
    checkpoint_path: str,
    output_path: str = "generated_painting.gif",
    n_samples: int = 1,
    sort_by_area: bool = True
):
    # Load model
    print("Loading model...")
    model = HeavyStrokeDiT(
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load renderer
    print("Loading renderer...")
    try:
        renderer = load_renderer(RENDERER_PATH, DEVICE)
        print("✓ Renderer loaded")
    except:
        print("⚠ Renderer not found")
        return
    
    # Generate strokes
    scheduler = XStartScheduler(num_steps=DIFFUSION_STEPS, device=DEVICE)
    generated_data = scheduler.sample(model, n_samples=n_samples, seq_len=SEQ_LEN, feat_dim=FEATURE_DIM)
    
    # Normalize to [0, 1]
    generated_data = torch.clamp((generated_data + 1) / 2, 0, 1)
    strokes = generated_data.view(n_samples, -1, 13).squeeze(0)  # (100, 13)
    
    if sort_by_area:
        print("Sorting strokes by scale...")
        strokes = sort_strokes_by_area(strokes)
    
    # Render
    canvas = torch.ones(1, 3, 128, 128).to(DEVICE)
    params, colors = strokes[:, :10], strokes[:, 10:]
    
    frames = []
    print("Rendering strokes...")
    
    with torch.no_grad():
        alphas_map = renderer(params)
        
        # Initial white canvas
        img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray((img * 255).astype(np.uint8)))
        
        for k in range(TOTAL_STROKES):
            canvas = canvas * (1 - alphas_map[k]) + colors[k].view(3, 1, 1) * alphas_map[k]
            img = canvas.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray((img * 255).astype(np.uint8)))
    
    # Save GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"✓ Saved: {output_path}")
    
    # Also save final frame
    final_path = output_path.replace('.gif', '_final.png')
    frames[-1].save(final_path)
    print(f"✓ Saved: {final_path}")
    
    # Display
    plt.imshow(frames[-1])
    plt.title("Generated Painting")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="generated_painting.gif")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--no_sort", action="store_true", help="Disable stroke sorting")
    args = parser.parse_args()
    
    generate(args.checkpoint, args.output, args.n_samples, not args.no_sort)
