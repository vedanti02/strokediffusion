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
        vgg = models.vgg16(pretrained=True).features[:layers].eval()
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


class CombinedLoss(nn.Module):
    """
    Combined loss for stroke-based painting training.
    Includes MSE on strokes + perceptual loss on rendered images.
    """
    
    def __init__(self, perceptual_weight: float = 0.25):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        pred_strokes: torch.Tensor,
        target_strokes: torch.Tensor,
        pred_image: torch.Tensor,
        target_image: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred_strokes: Predicted strokes
            target_strokes: Ground truth strokes
            pred_image: Rendered image from pred_strokes
            target_image: Rendered image from target_strokes
        Returns:
            Total loss and dict of individual losses
        """
        loss_mse = F.mse_loss(pred_strokes, target_strokes)
        loss_perceptual = self.perceptual(pred_image, target_image)
        
        total = loss_mse + self.perceptual_weight * loss_perceptual
        
        return total, {
            'mse': loss_mse.item(),
            'perceptual': loss_perceptual.item(),
            'total': total.item()
        }
