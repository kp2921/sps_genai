from fastapi import APIRouter
import torch
import os
from torchvision.utils import save_image
from helper_lib.model import get_model
from helper_lib.trainer import load_diffusion_checkpoint
from helper_lib.utils import get_device

router = APIRouter()

@router.get("/predict_diffusion")
async def predict_diffusion(num_samples: int = 8, steps: int = 100):
    """
    Generate images using the EMA diffusion model.
    Saves a grid to generated_diffusion/samples.png and returns the path.
    """
    device = get_device()
    model = get_model("diffusion").to(device)

    # Prefer best checkpoint if present, else last epoch
    ckpt_dir = "checkpoint/diffusion"
    best = os.path.join(ckpt_dir, "diffusion_best.pth")
    latest = None
    if os.path.isdir(ckpt_dir):
        epochs = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("diffusion_epoch_") and f.endswith(".pth")])
        latest = os.path.join(ckpt_dir, epochs[-1]) if epochs else None

    ckpt_path = best if os.path.exists(best) else latest
    if ckpt_path is None:
        return {"error": "No diffusion checkpoints found. Train with train_diffusion.py first."}

    load_diffusion_checkpoint(model, optimizer=None, checkpoint_path=ckpt_path, device=device)
    model.eval()

    os.makedirs("generated_diffusion", exist_ok=True)
    with torch.no_grad():
        imgs = model.generate(num_images=num_samples, diffusion_steps=steps, image_size=32)
    out_path = os.path.join("generated_diffusion", "samples.png")
    save_image(imgs, out_path, nrow=min(num_samples, 8))

    return {"num_generated": num_samples, "file": out_path, "checkpoint": ckpt_path}
