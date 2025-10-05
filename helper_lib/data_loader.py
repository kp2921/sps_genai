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
