import torch
from tqdm import tqdm
from helper_lib.checkpoints import save_checkpoint
from helper_lib.evaluator import evaluate_model

def train_model(model, train_loader, val_loader=None, criterion=None, optimizer=None,
                device='cpu', epochs=10, checkpoint_dir='checkpoint'):  # ✅ singular

    best_accuracy = 0.0
    model.to(device)

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct, running_total = 0, 0

        # Training loop with tqdm progress bar
        progress_bar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

            progress_bar.set_postfix({
                "loss": f"{running_loss / (running_total / labels.size(0)):.4f}",
                "acc": f"{running_correct / running_total:.3f}"
            })

        # Compute epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        print(f"\nEpoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

        # --- Optional validation ---
        if val_loader is not None:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {val_acc:.2f}%")

        # ✅ Save checkpoint in the "checkpoint" folder
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
            checkpoint_dir=checkpoint_dir
        )
        print(f"Checkpoint saved: {checkpoint_path}")

        # ✅ Save best model in a subfolder "checkpoint/best"
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best"
            )
            print(f"New best model saved at epoch {epoch+1} (Accuracy: {epoch_accuracy:.2f}%)")

    print("Training complete.")
    return model
