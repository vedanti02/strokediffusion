"""
Central configuration for all experiments.
Modify these values to change model architecture, training params, etc.
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
DATA_PATH = "/content/drive/MyDrive/LearningToPaint/strokes_fixed"
RENDERER_PATH = "renderer.pkl"
CHECKPOINT_DIR = "checkpoints"

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
SEQ_LEN = 20           # Number of stroke bundles (each bundle = 5 strokes)
FEATURE_DIM = 65       # Features per bundle (5 strokes * 13 params)
TOTAL_STROKES = 100    # Total strokes per painting (SEQ_LEN * 5)

D_MODEL = 768          # Transformer hidden dimension
NUM_HEADS = 12         # Attention heads
NUM_LAYERS = 12        # Transformer layers
MLP_RATIO = 4.0        # MLP expansion ratio

# ==========================================
# TRAINING
# ==========================================
BATCH_SIZE = 48
LEARNING_RATE = 1e-4
EPOCHS = 500
NUM_WORKERS = 4

# ==========================================
# DIFFUSION
# ==========================================
DIFFUSION_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# ==========================================
# LOSS WEIGHTS
# ==========================================
PERCEPTUAL_WEIGHT = 0.25
STROKE_WEIGHT = 1.0
CFG_DROP_PROB = 0.1    # Classifier-free guidance dropout

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
