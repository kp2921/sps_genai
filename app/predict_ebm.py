from fastapi import APIRouter
import torch
from helper_lib.model import get_model
from helper_lib.trainer import generate_samples
from helper_lib.utils import clip_img
from torchvision.utils import save_image
import os

router = APIRouter()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "checkpoint/ebm/model_epoch_final.pth"
os.makedirs("generated_ebm", exist_ok=True)

nn_energy_model = get_model("ebm").to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
nn_energy_model.load_state_dict(checkpoint["model_state_dict"])
nn_energy_model.eval()

@router.get("/predict_ebm")
async def generate_ebm_samples(num_samples: int = 8, steps: int = 60,
                               step_size: float = 10.0, noise_std: float = 0.005):
    """
    Generate samples using the trained EBM via Langevin dynamics.
    """
    inp_imgs = torch.rand((num_samples, 3, 32, 32), device=device) * 2 - 1

    with torch.no_grad():
        samples = generate_samples(nn_energy_model, inp_imgs,
                                   steps=steps, step_size=step_size, noise_std=noise_std)
        samples = clip_img(samples)
        file_paths = []
        for i in range(num_samples):
            out_path = os.path.join("generated_ebm", f"ebm_sample_{i+1}.png")
            save_image(samples[i], out_path)
            file_paths.append(out_path)

    return {"num_generated": num_samples, "files": file_paths}
