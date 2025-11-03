import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # input: 64x64x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Helper function: load the trained model
def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained SimpleCNN model from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    return model

# GAN Models (MNIST 28×28)

class Generator(nn.Module):
    """
    Generator for 28×28 grayscale (MNIST) images.
    Architecture:
      • FC → reshape → ConvT(128→64) → ConvT(64→1)
    """
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        # Fully connected layer: 100 → 7×7×128
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.bn0 = nn.BatchNorm1d(7 * 7 * 128)
        self.act0 = nn.ReLU(True)

        # ConvTranspose2D: 128 → 64 → 14×14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)

        # ConvTranspose2D: 64 → 1 → 28×28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Input z: (B, 100)
        x = self.fc(z)
        x = self.bn0(x)
        x = self.act0(x)
        x = x.view(z.size(0), 128, 7, 7)  # reshape to (B,128,7,7)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.tanh(x)
        return x

# Discriminator

class Discriminator(nn.Module):
    """
    Discriminator for 28×28 grayscale (MNIST) images.
    Architecture:
      • Conv(1→64) → Conv(64→128) → Linear(128×7×7→1)
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # First convolution: 1 → 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # Second convolution: 64 → 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Fully connected layer to output a single probability
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x: (B, 1, 28, 28)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# ============================================================
# Energy-Based Model (CIFAR-10)
# ============================================================

def swish(x):
    """Swish activation function (from slides)."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """
    Energy-Based Model using Swish activation (Module 8, CIFAR-10 version)
    --------------------------------------------------------------------
    Input: 3×32×32 (CIFAR-10 RGB)
    Architecture: 4 conv layers → 2 FC layers → scalar energy
    Output: scalar energy (lower = more data-like)
    """
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

# ============================================================
# Diffusion Model (CIFAR-10)
# ============================================================  

class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, t: torch.Tensor):
        t = t.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * t)
        cos_part = torch.cos(self.angular_speeds * t)
        out = torch.cat([sin_part, cos_part], dim=-1)
        return out.permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = (in_channels != out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if self.needs_projection else nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        h = self.act(self.bn1(x))
        h = self.conv1(h)
        h = self.act(self.bn2(h))
        h = self.conv2(h)
        return self.proj(x) + h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        blocks, ch = [], in_channels
        for _ in range(block_depth):
            blocks.append(ResidualBlock(ch, out_channels))
            ch = out_channels
        self.res = nn.Sequential(*blocks)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, block_depth=2):
        super().__init__()
        ch = in_channels + skip_channels
        blocks = []
        for _ in range(block_depth):
            blocks.append(ResidualBlock(ch, out_channels))
            ch = out_channels
        self.res = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class UNet(nn.Module):
    """
    U-Net for 32×32 RGB diffusion.
    Channel progression:
      down:  3→32→64→128→128
      mid:   128→256→256
      up:    (256+128)→128 → (128+128)→64 → (64+64)→32
    """
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, 1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(32, 32, 1)

        # Encoder
        self.down1 = DownBlock(32, 64)    # s1: 64
        self.down2 = DownBlock(64, 128)   # s2: 128
        self.down3 = DownBlock(128, 128)  # s3: 128

        # Bottleneck
        self.mid1 = ResidualBlock(128, 256)
        self.mid2 = ResidualBlock(256, 256)

        # Decoder
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up3 = UpBlock(64, 64, 32)

        self.final = nn.Conv2d(32, num_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x, t_embed):
        te = self.embedding_proj(self.embedding(t_embed))
        x0 = self.initial(x) + te

        d1, s1 = self.down1(x0)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)

        m = self.mid1(d3)
        m = self.mid2(m)

        u1 = self.up1(m, s3)
        u2 = self.up2(u1, s2)
        u3 = self.up3(u2, s1)

        return self.final(u3)


class DiffusionWrapper(nn.Module):
    def __init__(self, image_size=32, num_channels=3, schedule="cosine"):
        super().__init__()
        self.network = UNet(image_size=image_size, num_channels=num_channels)
        self.ema_network = UNet(image_size=image_size, num_channels=num_channels)
        self.ema_decay = 0.999
        self.register_buffer("normalizer_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("normalizer_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self._set_schedule(schedule)

    def _set_schedule(self, schedule): self.schedule = schedule

    def cosine_diffusion_schedule(self, diffusion_times):
        signal_rates = torch.cos(diffusion_times * math.pi / 2)
        noise_rates = torch.sin(diffusion_times * math.pi / 2)
        return noise_rates, signal_rates

    def _schedule_fn(self, diffusion_times):
        if self.schedule == "cosine":
            return self.cosine_diffusion_schedule(diffusion_times)
        return self.cosine_diffusion_schedule(diffusion_times)

    def update_ema(self):
        with torch.no_grad():
            for p, q in zip(self.ema_network.parameters(), self.network.parameters()):
                p.data.mul_(self.ema_decay).add_(q.data, alpha=1 - self.ema_decay)
    
    def set_normalizer(self, mean, std):
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)
        self.register_buffer("normalizer_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("normalizer_std", std.view(1, 3, 1, 1))

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        net = self.network if training else self.ema_network
        pred_noises = net(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps=100):
        step_size = 1.0 / diffusion_steps
        x = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((x.size(0), 1, 1, 1), device=x.device) * (1 - step * step_size)
            noise_rates, signal_rates = self._schedule_fn(t)
            pred_noises, pred_images = self.denoise(x, noise_rates, signal_rates, training=False)
            x = signal_rates * pred_images + noise_rates * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=100, image_size=32):
        z = torch.randn((num_images, 3, image_size, image_size), device=next(self.parameters()).device)
        imgs = self.reverse_diffusion(z, diffusion_steps)
        return torch.clamp(imgs * 0.5 + 0.5, 0, 1)

# ============================================================
# get_model selector
# ============================================================
def get_model(model_name: str):
    name = model_name.lower()
    if name == "cnn":
        return SimpleCNN()
    elif name == "generator":
        return Generator()
    elif name == "discriminator":
        return Discriminator()
    elif name == "ebm":
        return EnergyModel()
    elif name == "diffusion":
        return DiffusionWrapper(image_size=32, num_channels=3, schedule="cosine")
    else:
        raise ValueError(f"Unknown model name: {model_name}")