import torch
from torchvision.utils import save_image
from fastapi import File, UploadFile
from helper_lib.model import Generator
import os

# ---------------------------------------------------
# GAN PREDICTION / GENERATION MODULE
# ---------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your trained generator checkpoint
GEN_PATH = "checkpoint/generator/model_epoch_020.pth"

# Initialize the Generator model correctly
Z_DIM = 100
generator = Generator(z_dim=Z_DIM).to(device)

# Load checkpoint safely
checkpoint = torch.load(GEN_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    generator.load_state_dict(checkpoint["model_state_dict"])
else:
    generator.load_state_dict(checkpoint)

generator.eval()
print(f" Loaded GAN generator from: {GEN_PATH}")

# Output directory for generated images
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def generate_gan_image(num_images: int = 1):
    """
    Generate synthetic images using the trained GAN generator.
    Returns list of file paths to generated images.
    """
    with torch.no_grad():
        z = torch.randn(num_images, Z_DIM, device=device)
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2  # normalize to [0,1]

        file_paths = []
        for i in range(num_images):
            out_path = os.path.join(OUTPUT_DIR, f"generated_{i+1}.png")
            save_image(fake_images[i], out_path)
            file_paths.append(out_path)

    return {"num_generated": num_images, "files": file_paths}

