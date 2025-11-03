import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed set to: {seed}")

# ------------------------------------------------------------
# Clip and plot sample images (needed by train_ebm.py)
# ------------------------------------------------------------
@torch.no_grad()
def clip_img(x):
    return torch.clamp((x + 1) / 2, 0, 1)

@torch.no_grad()
def plot_samples(samples, n=8):
    samples = clip_img(samples)
    samples = samples.cpu()
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        img = samples[i].permute(1, 2, 0).squeeze()
        if img.ndim == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


