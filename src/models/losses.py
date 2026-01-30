"""
Loss functions for stroke-based painting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for comparing rendered images.
    Uses early VGG16 features for texture/structure comparison.
    """
    
    def __init__(self, layers: int = 16):
        super().__init__()
        vgg = models.vgg16(weights='IMAGENET1K_V1').features[:layers].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]
        Returns:
            Perceptual loss scalar
        """
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.mse_loss(self.vgg(pred), self.vgg(target))
