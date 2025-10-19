import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt

from helper_lib.model import Generator, Discriminator
from helper_lib.data_loader import get_mnist_loader      # <-- MNIST for GAN assignment
from helper_lib.trainer import train_gan
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import get_device, ensure_dir, set_seed

# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 20
BATCH_SIZE = 64
Z_DIM = 100
LR = 0.0002
BETA1 = 0.5

# Directory setup
BASE_DIR = "checkpoint"
SAMPLE_DIR = os.path.join(BASE_DIR, "gan_samples")
ensure_dir(BASE_DIR)
ensure_dir(SAMPLE_DIR)


# -------------------------------
# Setup Environment
# -------------------------------
set_seed(42)
device = get_device()

# Load MNIST data
train_loader, _ = get_mnist_loader(batch_size=BATCH_SIZE)


# -------------------------------
# Initialize Models & Optimizers
# -------------------------------
generator = Generator(Z_DIM).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))


# -------------------------------
# Visualization Helper
# -------------------------------
def save_gan_samples(generator, z_dim, epoch, device, sample_dir=SAMPLE_DIR):
    """
    Generates and saves GAN image samples for visual progress tracking.
    """
    os.makedirs(sample_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        # Fixed random noise for consistent outputs
        fixed_noise = torch.randn(16, z_dim, device=device)
        fake_images = generator(fixed_noise).detach().cpu()
        fake_images = (fake_images + 1) / 2  # rescale [-1,1] → [0,1]

        # Make a grid of 4x4 images
        grid = vutils.make_grid(fake_images, nrow=4)
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.imshow(grid.permute(1, 2, 0))
        path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    print(f"Saved GAN sample grid → {path}")


# -------------------------------
# GAN Training Loop
# -------------------------------
def run_gan_training():
    """
    Wrapper function that runs the GAN training and visualizer.
    """
    print(" Starting GAN training with automatic visualizer...")

    # Train GAN (uses trainer.py)
    train_gan(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        criterion=criterion,
        dataloader=train_loader,
        z_dim=Z_DIM,
        device=device,
        epochs=EPOCHS,
        checkpoint_dir=BASE_DIR
    )

    # Generate visual samples every 5 epochs
    for epoch in range(5, EPOCHS + 1, 5):
        save_gan_samples(generator, Z_DIM, epoch, device)

    print(" GAN training complete with visual samples saved!")


# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    run_gan_training()

