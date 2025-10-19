import os
import torch
import torch.nn as nn
from tqdm import tqdm
from helper_lib.checkpoints import save_checkpoint
from helper_lib.evaluator import evaluate_model


# ------------------------------------------------------------
# 🔹 CNN Training Loop 
# ------------------------------------------------------------
def train_cnn(model, train_loader, val_loader=None, criterion=None, optimizer=None,
              device='cpu', epochs=10, checkpoint_dir='checkpoint'):

    best_accuracy = 0.0
    model.to(device)

    print(f"Starting CNN training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct, running_total = 0, 0

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

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        print(f"\nEpoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

        if val_loader is not None:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save checkpoint
        checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                                        checkpoint_dir=os.path.join(checkpoint_dir, 'best'))
            print(f"New best model saved at epoch {epoch+1} (Accuracy: {epoch_accuracy:.2f}%)")

    print("CNN training complete.")
    return model


# ------------------------------------------------------------
# 🔹 GAN Checkpoint Helper
# ------------------------------------------------------------
def save_gan_checkpoint(generator, discriminator,
                        g_optimizer, d_optimizer,
                        epoch, g_loss, d_loss,
                        checkpoint_dir='checkpoint/gan'):
    """
    Save both Generator and Discriminator checkpoints.
    Each epoch, saves:
        - generator/model_epoch_XXX.pth
        - discriminator/model_epoch_XXX.pth
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    gen_dir = os.path.join(checkpoint_dir, "generator")
    disc_dir = os.path.join(checkpoint_dir, "discriminator")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(disc_dir, exist_ok=True)

    # Save generator
    gen_path = save_checkpoint(
        generator, g_optimizer, epoch, g_loss, 0, checkpoint_dir=gen_dir
    )

    # Save discriminator
    disc_path = save_checkpoint(
        discriminator, d_optimizer, epoch, d_loss, 0, checkpoint_dir=disc_dir
    )

    print(f"Checkpoint saved for epoch {epoch}:")
    print(f"  Generator → {gen_path}")
    print(f"  Discriminator → {disc_path}")


# ------------------------------------------------------------
# 🔹 GAN Training Loop
# ------------------------------------------------------------
def train_gan(generator, discriminator, g_optimizer, d_optimizer,
              criterion, dataloader, z_dim=100, device='cpu', epochs=20,
              checkpoint_dir='checkpoint/gan'):
    """
    Standard GAN training loop for MNIST dataset.
    """

    print(f"Starting GAN training for {epochs} epochs on {device}...")

    generator.to(device)
    discriminator.to(device)

    best_g_loss = float('inf')

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        g_loss_total, d_loss_total = 0.0, 0.0

        progress_bar = tqdm(dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")

        for real_imgs, _ in progress_bar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # ------------------------------------------------
            # Train Discriminator
            # ------------------------------------------------
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = generator(noise)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d_optimizer.zero_grad()

            real_outputs = discriminator(real_imgs)
            fake_outputs = discriminator(fake_imgs.detach())

            d_real_loss = criterion(real_outputs, real_labels)
            d_fake_loss = criterion(fake_outputs, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            # ------------------------------------------------
            # Train Generator
            # ------------------------------------------------
            g_optimizer.zero_grad()
            fake_outputs = discriminator(fake_imgs)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            progress_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}"
            })

        print(f"\nEpoch {epoch+1}: D_loss={d_loss_total/len(dataloader):.4f}, "
              f"G_loss={g_loss_total/len(dataloader):.4f}")

        # Save checkpoints each epoch
        save_gan_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                            epoch + 1, g_loss_total, d_loss_total, checkpoint_dir)

        # Save best generator (lowest G_loss)
        if g_loss_total < best_g_loss:
            best_g_loss = g_loss_total
            save_gan_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                                epoch + 1, g_loss_total, d_loss_total,
                                checkpoint_dir=os.path.join(checkpoint_dir, "best"))
            print(f"New best Generator saved (epoch {epoch+1}, loss={g_loss_total:.4f})")

    print("GAN training complete.")
