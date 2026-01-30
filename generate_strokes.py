"""
Generate stroke parameters from CelebA dataset.
Converts images to stroke sequences using the Learning to Paint baseline.

Usage:
    # Download CelebA dataset and generate strokes
    python generate_strokes.py --download_dataset --download_models
    
    # Or specify custom image directory
    python generate_strokes.py --img_dir /path/to/celeba/images

Outputs (in baseline/ directory):
    - baseline/output_pts/          : Stroke parameter .pt files (20, 65) -> reshape to (100, 13)
"""
import os
import shutil
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

# Default dataset location
DEFAULT_DATASET_DIR = "/home/vkshirsa/celeba_temp"


def setup_kaggle_credentials():
    """
    Set up Kaggle credentials from the strokediffusion directory.
    Copies kaggle.json to ~/.kaggle/ if not already there.
    """
    kaggle_dest = Path.home() / ".kaggle" / "kaggle.json"
    
    # If already set up, we're good
    if kaggle_dest.exists():
        return True
    
    # Look for kaggle.json in the script directory (strokediffusion/)
    script_dir = Path(__file__).parent
    kaggle_src = script_dir / "kaggle.json"
    
    if kaggle_src.exists():
        print(f"Found kaggle.json in {script_dir}")
        kaggle_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy(kaggle_src, kaggle_dest)
        
        # Set permissions (read/write for owner only)
        os.chmod(kaggle_dest, 0o600)
        print(f"[OK] Copied kaggle.json to {kaggle_dest}")
        return True
    
    return False


# def download_celeba_dataset(dataset_dir: str = DEFAULT_DATASET_DIR):
#     """
#     Download CelebA dataset from Kaggle.
    
#     Args:
#         dataset_dir: Directory to download dataset to
        
#     Returns:
#         Path to the image directory
#     """
#     dataset_dir = Path(dataset_dir)
#     dataset_dir.mkdir(parents=True, exist_ok=True)
    
#     celeba_dir = dataset_dir / "celeba"
#     img_dir = celeba_dir / "img_align_celeba" / "img_align_celeba"
    
#     # Check if already downloaded
#     if img_dir.exists() and any(img_dir.glob("*.jpg")):
#         print(f"[OK] CelebA dataset already exists at {img_dir}")
#         return str(img_dir)
    
#     print(f"Downloading CelebA dataset to {dataset_dir}...")
    
#     # Check if kaggle is installed
#     try:
#         subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         print("Installing kaggle CLI...")
#         subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    
#     # Set up kaggle credentials from strokediffusion directory
#     setup_kaggle_credentials()
    
#     # Check for kaggle credentials
#     kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
#     if not kaggle_json.exists():
#         script_dir = Path(__file__).parent
#         print("\n[WARN] Kaggle API key not found!")
#         print(f"Please place kaggle.json in: {script_dir}/")
#         print("\nTo get kaggle.json:")
#         print("  1. Go to https://www.kaggle.com/account")
#         print("  2. Click 'Create New API Token' to download kaggle.json")
#         print(f"  3. Copy it to: {script_dir}/kaggle.json")
#         raise FileNotFoundError(f"Kaggle API key not found. Place kaggle.json in {script_dir}/")
    
#     # Download dataset
#     zip_path = dataset_dir / "celeba-dataset.zip"
#     if not zip_path.exists():
#         print("Downloading from Kaggle (this may take a while, ~1.3GB)...")
#         subprocess.run(
#             ["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset", "-p", str(dataset_dir)],
#             check=True
#         )
    
#     # Extract
#     if not celeba_dir.exists():
#         print("Extracting dataset...")
#         subprocess.run(
#             ["unzip", "-q", str(zip_path), "-d", str(celeba_dir)],
#             check=True
#         )
    
#     # Clean up zip file (optional, comment out to keep)
#     zip_path.unlink()
    
#     print(f"CelebA dataset downloaded to {img_dir}")
#     return str(img_dir)


def download_models(model_dir: str = "."):
    """Download pretrained actor and renderer models."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    actor_path = model_dir / "actor.pkl"
    renderer_path = model_dir / "renderer.pkl"
    
    # actor.pkl download
    if not actor_path.exists():
        print("Downloading actor.pkl...")
        try:
            # Try gdown first (handles Google Drive better)
            subprocess.run(
                ["pip", "install", "-q", "gdown"],
                capture_output=True
            )
            subprocess.run([
                "gdown", "--id", "1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR",
                "-O", str(actor_path)
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to wget
            subprocess.run([
                "wget", "-q", "--show-progress",
                "https://drive.google.com/uc?export=download&id=1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR",
                "-O", str(actor_path)
            ], check=True)
        print(f"[OK] Downloaded actor.pkl to {actor_path}")
    else:
        print(f"[OK] actor.pkl already exists at {actor_path}")
    
    # renderer.pkl download
    if not renderer_path.exists():
        print("Downloading renderer.pkl...")
        try:
            subprocess.run([
                "gdown", "--id", "1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4",
                "-O", str(renderer_path)
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run([
                "wget", "-q", "--show-progress",
                "https://drive.google.com/uc?export=download&id=1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4",
                "-O", str(renderer_path)
            ], check=True)
        print(f"[OK] Downloaded renderer.pkl to {renderer_path}")
    else:
        print(f"[OK] renderer.pkl already exists at {renderer_path}")
    
    return str(actor_path), str(renderer_path)


def get_image_list(img_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')):
    """Get list of image files from directory."""
    img_dir = Path(img_dir)
    images = []
    for ext in extensions:
        images.extend(img_dir.glob(f"*{ext}"))
        images.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(images)


def generate_strokes(
    img_dir: str,
    actor_path: str = "actor.pkl",
    renderer_path: str = "renderer.pkl",
    max_step: int = 20,
    divide: int = 1,
    start_idx: int = 0,
    end_idx: int = None,
    skip_existing: bool = True,
    working_dir: str = None
):
    """
    Generate stroke parameters for images in a directory.
    
    Args:
        img_dir: Directory containing input images
        actor_path: Path to actor.pkl model
        renderer_path: Path to renderer.pkl model
        max_step: Maximum steps for stroke generation (default: 20)
        divide: Division parameter for test.py (default: 1)
        start_idx: Start index in image list (for resuming)
        end_idx: End index in image list (None = all)
        skip_existing: Skip images that already have output files
        working_dir: Directory to run test.py from (default: script directory)
    
    Outputs are saved to:
        - output_pts/: Stroke parameter .pt files
    """
    # Verify paths exist
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor model not found: {actor_path}")
    if not os.path.exists(renderer_path):
        raise FileNotFoundError(f"Renderer model not found: {renderer_path}")
    
    # Get absolute paths
    img_dir = os.path.abspath(img_dir)
    actor_path = os.path.abspath(actor_path)
    renderer_path = os.path.abspath(renderer_path)
    
    # Set working directory to baseline/ where test.py creates outputs
    if working_dir is None:
        working_dir = Path(__file__).parent / "baseline"
    working_dir = Path(working_dir)
    
    # Output directories (created by test.py in its working directory)
    output_pts_dir = working_dir / "output_pts"
    
    # Create output directories
    os.makedirs(output_pts_dir, exist_ok=True)
    
    # Get image list
    images = get_image_list(img_dir)
    if not images:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(images)} images in {img_dir}")
    
    # Slice image list
    if end_idx is None:
        end_idx = len(images)
    images = images[start_idx:end_idx]
    print(f"Processing images {start_idx} to {end_idx} ({len(images)} images)")
    
    # Get test.py path (in baseline/ directory which is now working_dir)
    test_script = working_dir / "test.py"
    if not test_script.exists():
        raise FileNotFoundError(f"test.py not found at {test_script}")
    
    # Process images
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for img_path in tqdm(images, desc="Generating strokes"):
        img_name = img_path.stem
        output_file = output_pts_dir / f"{img_name}.pt"
        
        # Skip if output exists
        if skip_existing and output_file.exists():
            skip_count += 1
            continue
        
        try:
            result = subprocess.run(
                [
                    "python3", str(test_script),
                    f"--max_step={max_step}",
                    f"--actor={actor_path}",
                    f"--renderer={renderer_path}",
                    f"--img={img_path}",
                    f"--divide={divide}"
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(working_dir)
            )
            success_count += 1
        except subprocess.CalledProcessError as e:
            error_count += 1
            tqdm.write(f"Error processing {img_path.name}: {e.stderr[:200] if e.stderr else 'Unknown error'}")
    
    print(f"\n[OK] Generation complete!")
    print(f"  - Processed: {success_count}")
    print(f"  - Skipped (existing): {skip_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Stroke files: {output_pts_dir}")
    print(f"\nNote: Each .pt file is shape (20, 65). Reshape to (100, 13) for model input:")
    print(f"  strokes = torch.load('file.pt').view(-1, 13)  # -> (100, 13)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stroke parameters from images using Learning to Paint baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Download everything and generate strokes (recommended first run)
    python generate_strokes.py --download_dataset --download_models
    
    # Generate strokes for custom image directory
    python generate_strokes.py --img_dir ./my_images
    
    # Resume from a specific index
    python generate_strokes.py --img_dir ./celeba/images --start_idx 10956
    
    # Process a specific range
    python generate_strokes.py --img_dir ./celeba/images --start_idx 0 --end_idx 1000

Default paths:
    - Dataset: {DEFAULT_DATASET_DIR}/celeba/img_align_celeba/img_align_celeba
    - Models: ./actor.pkl, ./renderer.pkl

Output directories (created in baseline/):
    - baseline/output_pts/           : Stroke parameter .pt files
        """
    )
    
    parser.add_argument("--img_dir", type=str, help="Directory containing input images (default: CelebA location)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, 
                        help=f"Directory to download CelebA dataset (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--actor", type=str, default="actor.pkl", help="Path to actor.pkl model")
    parser.add_argument("--renderer", type=str, default="renderer.pkl", help="Path to renderer.pkl model")
    parser.add_argument("--max_step", type=int, default=20, help="Maximum steps for stroke generation (20 steps = 100 strokes)")
    parser.add_argument("--divide", type=int, default=1, help="Division parameter for higher resolution (1=128x128)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in image list (for resuming)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index in image list (default: all)")
    parser.add_argument("--no_skip", action="store_true", help="Don't skip existing output files (reprocess all)")
    parser.add_argument("--download_models", action="store_true", help="Download pretrained actor and renderer models")
    parser.add_argument("--download_dataset", action="store_true", help="Download CelebA dataset from Kaggle")
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        args.actor, args.renderer = download_models(".")
    
    # Download dataset if requested
    if args.download_dataset:
        args.img_dir = download_celeba_dataset(args.dataset_dir)
    
    # If no img_dir specified, try default CelebA location
    if not args.img_dir:
        default_img_dir = Path(args.dataset_dir) / "celeba" / "img_align_celeba" / "img_align_celeba"
        if default_img_dir.exists():
            args.img_dir = str(default_img_dir)
            print(f"Using default CelebA location: {args.img_dir}")
        else:
            parser.error("--img_dir is required (or use --download_dataset to download CelebA)")
    
    # Check if models exist, offer to download
    if not os.path.exists(args.actor) or not os.path.exists(args.renderer):
        print("Models not found. Downloading...")
        args.actor, args.renderer = download_models(".")
    
    # Generate strokes
    generate_strokes(
        img_dir=args.img_dir,
        actor_path=args.actor,
        renderer_path=args.renderer,
        max_step=args.max_step,
        divide=args.divide,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_existing=not args.no_skip
    )


if __name__ == "__main__":
    main()
