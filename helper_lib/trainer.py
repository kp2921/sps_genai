import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict

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

# ============================================================
# Metric Tracker
# ============================================================
class Metric:
    def __init__(self):
        self.reset()
    def update(self, val): self.total += val.item(); self.count += 1
    def result(self): return self.total / self.count if self.count > 0 else 0.0
    def reset(self): self.total, self.count = 0.0, 0


# ============================================================
# Langevin Dynamics Sampler (used in EBM)
# ============================================================
def generate_samples(nn_energy_model, inp_imgs, steps=60, step_size=10.0, noise_std=0.005):
    nn_energy_model.eval()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
        inp_imgs.requires_grad_(True)
        energy = nn_energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    return inp_imgs.detach()


# ============================================================
# Persistent Replay Buffer
# ============================================================
class Buffer:
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.examples = [torch.rand((1, 3, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise_std):
        n_new = np.random.binomial(128, 0.05)
        new_rand_imgs = torch.rand((n_new, 3, 32, 32), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise_std)
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


# ============================================================
# Energy-Based Model Wrapper (Training Logic)
# ============================================================
class EBM(nn.Module):
    def __init__(self, model, alpha, steps, step_size, noise, device):
        super().__init__()
        self.model = model
        self.device = device
        self.buffer = Buffer(self.model, device=device)
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise
        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                  self.real_out_metric, self.fake_out_metric]:
            m.reset()

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = self.buffer.sample_new_exmps(
            steps=self.steps, step_size=self.step_size, noise_std=self.noise
        )

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)

        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())
        return self.metrics()

    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand((batch_size, 3, 32, 32), device=self.device) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        with torch.no_grad():
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
            cdiv = real_out.mean() - fake_out.mean()
        self.cdiv_loss_metric.update(cdiv)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())
        return {
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

# ============================================================
# Diffusion training
# ============================================================

def compute_channelwise_mean_std(dataloader, device="cpu", num_channels=3):
    mean = torch.zeros(num_channels, device=device)
    std = torch.zeros(num_channels, device=device)
    total = 0

    for imgs, _ in dataloader:
        imgs = imgs.to(device)  # shape: (B, C, H, W)
        b = imgs.size(0)
        mean += imgs.mean(dim=(0, 2, 3)) * b
        std  += imgs.std(dim=(0, 2, 3), unbiased=False) * b
        total += b

    mean /= total
    std  /= total

    return mean.view(1, num_channels, 1, 1), std.view(1, num_channels, 1, 1)

def train_diffusion(model, train_loader, val_loader, optimizer,
                    loss_fn=None, epochs=10, device="cpu",
                    checkpoint_dir="checkpoint/diffusion"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # (Optional) recompute normalization like in notebook
    with torch.no_grad():
        mean, std = compute_channelwise_mean_std(train_loader, device=device, num_channels=3)
        # If you prefer fixed (0.5,0.5,0.5), comment next line
        model.set_normalizer(mean, std)

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            imgs = imgs.to(device)
            noises = torch.randn_like(imgs)
            t = torch.rand((imgs.size(0),1,1,1), device=device)
            noise_rates, signal_rates = model._schedule_fn(t)
            noisy_imgs = signal_rates * imgs + noise_rates * noises

            pred_noises, _ = model.denoise(noisy_imgs, noise_rates, signal_rates, training=True)
            loss = loss_fn(pred_noises, noises)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_ema()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                imgs = imgs.to(device)
                noises = torch.randn_like(imgs)
                t = torch.rand((imgs.size(0),1,1,1), device=device)
                noise_rates, signal_rates = model._schedule_fn(t)
                noisy_imgs = signal_rates * imgs + noise_rates * noises
                pred_noises, _ = model.denoise(noisy_imgs, noise_rates, signal_rates, training=False)
                loss = loss_fn(pred_noises, noises)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        # save latest
        ckpt = {
            "epoch": epoch+1,
            "model_state_dict": model.network.state_dict(),
            "ema_model_state_dict": model.ema_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer_mean": model.normalizer_mean,
            "normalizer_std": model.normalizer_std,
        }
        latest_path = os.path.join(checkpoint_dir, f"diffusion_epoch_{epoch+1:03d}.pth")
        torch.save(ckpt, latest_path)
        print(f"💾 Checkpoint saved → {latest_path}")

        # best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(checkpoint_dir, "diffusion_best.pth")
            torch.save(ckpt, best_path)
            print(f"🌟 New best (val {val_loss:.4f}) saved → {best_path}")

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

def load_diffusion_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(ckpt["model_state_dict"])
    model.ema_network.load_state_dict(ckpt["ema_model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "normalizer_mean" in ckpt and "normalizer_std" in ckpt:
        model.set_normalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])

        
