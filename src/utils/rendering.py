"""
Utility functions for rendering strokes to canvas.
"""
import torch
from typing import Optional


def render_batch(
    params: torch.Tensor,
    renderer,
    canvas_size: int = 128,
    num_strokes: int = 100
) -> torch.Tensor:
    """
    Differentiable batch rendering of stroke parameters to images.
    
    Args:
        params: Stroke parameters of shape (B, L, F) in [-1, 1]
               where L * F / 13 = num_strokes
        renderer: Neural renderer module
        canvas_size: Output image size
        num_strokes: Number of strokes per image
    
    Returns:
        Rendered images of shape (B, 3, canvas_size, canvas_size)
    """
    B, L, _ = params.shape
    
    # Normalize from [-1, 1] to [0, 1] for renderer
    params = torch.clamp((params + 1) / 2, 0, 1)
    
    # Reshape to individual strokes
    strokes = params.view(B, -1, 13)
    shape_params = strokes[:, :, :10].reshape(-1, 10)
    color_params = strokes[:, :, 10:]
    
    # Get alpha maps for all strokes at once
    alphas = renderer(shape_params).view(B, num_strokes, 1, canvas_size, canvas_size)
    
    # Start with white canvas
    canvas = torch.ones(B, 3, canvas_size, canvas_size).to(params.device)
    colors = color_params.view(B, num_strokes, 3, 1, 1)
    
    # Alpha composite all strokes
    for i in range(num_strokes):
        canvas = canvas * (1 - alphas[:, i]) + colors[:, i] * alphas[:, i]
    
    return canvas


def render_strokes(
    strokes: torch.Tensor,
    renderer,
    canvas: Optional[torch.Tensor] = None,
    canvas_size: int = 128
) -> torch.Tensor:
    """
    Render a sequence of strokes to canvas (single image).
    
    Args:
        strokes: Stroke parameters of shape (N, 13) in [0, 1]
        renderer: Neural renderer module
        canvas: Optional starting canvas (defaults to white)
        canvas_size: Canvas size
    
    Returns:
        Rendered image of shape (1, 3, canvas_size, canvas_size)
    """
    device = strokes.device
    
    if canvas is None:
        canvas = torch.ones(1, 3, canvas_size, canvas_size).to(device)
    
    shape_params = strokes[:, :10]
    color_params = strokes[:, 10:]
    
    with torch.no_grad():
        alphas = renderer(shape_params)  # (N, 1, H, W)
        
        for i in range(strokes.shape[0]):
            alpha = alphas[i]
            color = color_params[i].view(3, 1, 1)
            canvas = canvas * (1 - alpha) + color * alpha
    
    return canvas


def sort_strokes_by_area(strokes: torch.Tensor) -> torch.Tensor:
    """
    Sort strokes by area (largest first) for better rendering.
    Large strokes (background) should be drawn first, small (details) last.
    
    Args:
        strokes: Stroke parameters of shape (N, 13)
                 Assumes indices 2, 3 are width, height
    
    Returns:
        Sorted strokes of shape (N, 13)
    """
    w = strokes[:, 2]
    h = strokes[:, 3]
    areas = w * h
    
    # Sort by area descending (largest first)
    sorted_indices = torch.argsort(areas, descending=True)
    return strokes[sorted_indices]
