"""
Central configuration for progressive stroke diffusion.
"""
import torch
import os

# ==========================================
# DEVICE
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# PATHS
# ==========================================
DATA_PATH = "./baseline/output_pts"
RENDERER_PATH = "renderer.pkl"
CHECKPOINT_DIR = "checkpoints"

# ==========================================
# STROKE REPRESENTATION
# ==========================================
# Each stroke has 13 parameters:
#   - Shape (10): x0, y0, x1, y1, x2, y2, radius0, radius1, w, h
#   - Color (3): R, G, B
NUM_STROKES = 100      # Total strokes per painting
STROKE_DIM = 13        # Parameters per stroke

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
D_MODEL = 384          # Transformer hidden dimension
NUM_HEADS = 8          # Attention heads
NUM_LAYERS = 8         # Transformer layers

# ==========================================
# TRAINING
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 300
NUM_WORKERS = 4

# ==========================================
# DIFFUSION
# ==========================================
DIFFUSION_STEPS = 1000

# ==========================================
# RENDERING
# ==========================================
CANVAS_SIZE = 128

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_save_dir(experiment_name: str) -> str:
    """Create and return save directory for an experiment."""
    save_dir = os.path.join(CHECKPOINT_DIR, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
