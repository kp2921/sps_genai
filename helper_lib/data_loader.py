import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size=32):
    transform = transforms.Compose([
    transforms.Resize((64, 64)),   # 🔹 resize before converting to tensor
    transforms.ToTensor()
])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_mnist_loader(batch_size=64):
    """
    Loads the MNIST dataset with normalization suitable for GAN training.
    Output range: [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalize grayscale to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded MNIST dataset | Batch size = {batch_size}")
    return train_loader, test_loader