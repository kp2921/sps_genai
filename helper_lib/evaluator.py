import torch
import torch.nn as nn

def evaluate_model(model, data_loader, device='cpu', criterion=None):

    model.eval()  # set model to evaluation mode
    total, correct = 0, 0
    running_loss = 0.0

    # Use no_grad for faster inference (no backprop)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader) if criterion is not None else None

    # Print summary
    if avg_loss is not None:
        print(f"Evaluation → Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    else:
        print(f"Evaluation → Accuracy: {accuracy:.2f}%")

    return accuracy, avg_loss
