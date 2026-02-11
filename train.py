"""
Training entry point — delegates to baseline/train.py

Usage:
    python train.py --data_path /path/to/strokes --image_path /path/to/images
    python train.py --epochs 500 --resume checkpoint.pth
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Progressive stroke diffusion training (CFG-enabled)")
    parser.add_argument("--data_path", type=str, default=None, help="Dir with stroke .pt files")
    parser.add_argument("--image_path", type=str, default=None, help="Dir with source face images")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(args.data_path, args.image_path, args.batch_size, args.lr, args.epochs, args.resume)