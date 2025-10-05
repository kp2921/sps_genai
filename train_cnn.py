from helper_lib.model import SimpleCNN
from helper_lib.data_loader import get_cifar10_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import get_device, ensure_dir, set_seed
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Setup environment
# -------------------------------
set_seed(42)
device = get_device()
ensure_dir("checkpoint")

# -------------------------------
# Load CIFAR-10 data
# -------------------------------
train_loader, test_loader = get_cifar10_loader(batch_size=32)

# -------------------------------
# Initialize model, loss, optimizer
# -------------------------------
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -------------------------------
# Train the model
# -------------------------------
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=5,
    checkpoint_dir="checkpoint"
)

# -------------------------------
# Evaluate final model
# -------------------------------
evaluate_model(trained_model, test_loader, device, criterion)
