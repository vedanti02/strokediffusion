import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# --- 1. Define the PixelShuffle Renderer ---
class NeuralRenderer(nn.Module):
    def __init__(self):
        super(NeuralRenderer, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        return torch.sigmoid(x)

def generate_face():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer_path = "renderer.pkl"
    stroke_path = "028768_new.pt"
    output_filename = "reconstructed_face.png"

    # 1. Load Renderer
    print("Loading Renderer...")
    renderer = NeuralRenderer().to(device)
    try:
        renderer.load_state_dict(torch.load(renderer_path, map_location=device))
        renderer.eval()
    except Exception as e:
        print(f"Error loading renderer: {e}")
        return

    # 2. Load Strokes
    print("Loading Strokes...")
    try:
        actions = torch.load(stroke_path, map_location=device)
        # Flatten if needed: (Steps, 65)
        if actions.dim() == 3: actions = actions.squeeze(0)
    except Exception as e:
        print(f"Error loading strokes: {e}")
        return

    # --- 3. AUTO-FIX SCALING ---
    print(f"Raw Action Range: Min={actions.min().item():.3f}, Max={actions.max().item():.3f}")

    # If the data is roughly -1 to 1, we must shift it to 0 to 1
    if actions.min() < -0.1:
        print("Detected [-1, 1] range. Normalizing to [0, 1]...")
        actions = (actions + 1) / 2
    else:
        print("Range seems okay (already 0-1). Keeping as is.")

    # 4. Render
    # White Canvas (1.0) looks better for faces than Black (0.0)
    canvas = torch.ones([1, 3, 128, 128]).to(device)

    FORCE_RANDOM_COLORS = False
    print("Painting...")
    with torch.no_grad():
        for i in range(actions.shape[0]):
            # Get bundle of 5 strokes (Params: 13 per stroke)
            bundle = actions[i].view(-1, 13) # Shape [5, 13]

            # Extract Shape Params (First 10) & Colors (Last 3)
            xy_params = bundle[:, :10]
            rgb_params = bundle[:, 10:]

            if i == 0:
                r, g, b = rgb_params[0].tolist()
                print(f"DEBUG: Stroke 1 Color Data -> R:{r:.2f} G:{g:.2f} B:{b:.2f}")
                if abs(r-g) < 0.05 and abs(g-b) < 0.05:
                    print(">>> DETECTED GRAYSCALE DATA (R, G, and B are nearly equal) <<<")
            # Generate Alpha Masks (Shapes)
            # Renderer outputs 1.0 for Ink, 0.0 for Empty
            alpha_maps = renderer(xy_params) # [5, 1, 128, 128]

            # Apply to Canvas
            for j in range(5):
                alpha = alpha_maps[j]
                color = rgb_params[j].view(3, 1, 1)

                # Standard Alpha Blending
                # Canvas = Canvas * (1 - Alpha) + Color * Alpha
                canvas = canvas * (1 - alpha) + color * alpha

            if i % 5 == 0: print(f"Step {i}/{actions.shape[0]}")

    # 5. Save
    img_np = canvas[0].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype('uint8')
    # Convert RGB to BGR for OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_filename, img_np)
    print(f"Success! Face saved to {output_filename}")

if __name__ == "__main__":
    generate_face()