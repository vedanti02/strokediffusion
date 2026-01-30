# StrokeDiffusion

Stroke-based image generation using Diffusion Transformers. This project combines reinforcement learning-based stroke generation with diffusion models to create painterly renderings of images.

## Overview

This project has two main components:

1. **Baseline (RL-based)**: Uses a trained actor network to generate stroke parameters from images
2. **Diffusion Model**: Learns to generate strokes using a Diffusion Transformer (DiT)

## Project Structure

```
strokediffusion/
├── baseline/                    # RL-based stroke generation
│   ├── DRL/                     # Deep RL components (actor, critic, etc.)
│   ├── Renderer/                # Neural stroke renderer
│   ├── test.py                  # Generate strokes from images
│   ├── train.py                 # Train RL actor
│   └── train_renderer.py        # Train renderer
├── src/                         # Diffusion model (modular)
│   ├── models/                  # DiT, Renderer, Losses
│   ├── data/                    # Dataset classes
│   ├── diffusion/               # Noise schedulers
│   └── utils/                   # Rendering utilities
├── train.py                     # Train diffusion model
├── test.py                      # Test diffusion model (image → strokes)
├── finetune.py                  # Fine-tune with MSE loss
├── generate.py                  # Unconditional generation
└── generate_image_from_strokes_general.py  # Render strokes to image
```

## Workflow

### 1. Generate Training Data (using RL baseline)

```bash
cd baseline
python test.py --img path/to/image.png --actor model/actor.pkl --renderer renderer.pkl
```

This saves stroke parameters as `.pt` files.

### 2. Train Diffusion Model

```bash
python train.py
```

Configure paths in `src/config.py`.

### 3. Generate Strokes with Diffusion

```bash
# Image-conditioned generation
python test.py --checkpoint checkpoints/model.pth --image path/to/image.png

# Unconditional generation
python generate.py --checkpoint checkpoints/model.pth
```

### 4. Render Strokes to Image

```bash
python generate_image_from_strokes_general.py
```

## Model Architecture

- **VisualStrokeDiT**: Image-conditioned Diffusion Transformer
  - 12 layers, 768 hidden dim, 12 attention heads
  - Image encoder for conditioning
  - Predicts clean strokes (x₀ prediction)

- **NeuralRenderer**: Converts 10D stroke params → 128×128 alpha mask
  - FC layers + PixelShuffle upsampling

## Requirements

```
torch
torchvision
numpy
opencv-python
matplotlib
Pillow
tqdm
```

## Citation

Based on "Learning to Paint" by Huang et al.
