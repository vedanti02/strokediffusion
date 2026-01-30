"""
Neural Renderer for stroke-based painting.
Converts 10-dimensional stroke parameters to alpha masks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralRenderer(nn.Module):
    """
    Fully-connected + convolutional neural renderer.
    Takes 10D stroke parameters and outputs a 128x128 alpha mask.
    """
    def __init__(self):
        super(NeuralRenderer, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        
        # Convolutional layers with PixelShuffle upsampling
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stroke parameters of shape (B, 10)
        Returns:
            Alpha mask of shape (B, 1, 128, 128)
        """
        # FC layers: (B, 10) -> (B, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        # Reshape to spatial: (B, 4096) -> (B, 16, 16, 16)
        x = x.view(-1, 16, 16, 16)
        
        # Upsample: 16x16 -> 32x32 -> 64x64 -> 128x128
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))  # 32x32
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))  # 64x64
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))  # 128x128
        
        return torch.sigmoid(x)


def load_renderer(path: str, device: str = "cuda") -> NeuralRenderer:
    """Load a pretrained renderer from disk."""
    renderer = NeuralRenderer().to(device)
    renderer.load_state_dict(torch.load(path, map_location=device))
    renderer.eval()
    for p in renderer.parameters():
        p.requires_grad = False
    return renderer
