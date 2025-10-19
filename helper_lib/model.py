import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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

