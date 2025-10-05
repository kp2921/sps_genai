import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoint'):
    """
    Save model checkpoint safely, creating the directory if needed.
    Works on Windows, macOS, and Linux.
    """
    # Normalize and ensure folder exists
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}  |  Epoch {epoch}  |  Accuracy: {accuracy:.2f}%")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """
    Load model checkpoint and restore training state.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('accuracy', None)

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"   → Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    return epoch, loss, accuracy
