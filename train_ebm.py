# ============================================================
# train_ebm.py — Energy-Based Model Training Script (CIFAR-10)
# Adaptive (CPU/GPU) version — full training + image generation
# ============================================================

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from helper_lib.model import get_model
    from helper_lib.trainer import EBM
    from helper_lib.utils import plot_samples, get_device, set_seed
    import matplotlib.pyplot as plt
    import os
    import sys

    # ------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------
    plt.ioff()  # disable interactive mode (no popup windows)
    set_seed(42)
    device = get_device()
    print(f"✅ Environment ready — using {device.upper()}")

    # ------------------------------------------------------------
    # 2. Dataset (CIFAR-10)
    # ------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(root="data", train=True, download=False, transform=transform)
    print(f"Loaded CIFAR-10 training set: {len(train_data)} samples")

    test_data = datasets.CIFAR10(root="data", train=False, download=False, transform=transform)
    print(f"Loaded CIFAR-10 test set: {len(test_data)} samples")

    print("Initializing DataLoaders...")
    batch_size = 32 if device == "cpu" else 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print("✅ DataLoaders initialized")

    # ------------------------------------------------------------
    # 3. Model & Optimizer
    # ------------------------------------------------------------
    nn_energy_model = get_model("ebm").to(device)

    # Adaptive config: CPU uses lighter settings
    steps = 20 if device == "cpu" else 60
    step_size = 6 if device == "cpu" else 10
    epochs = 8 if device == "cpu" else 10

    ebm = EBM(
        nn_energy_model,
        alpha=0.1,
        steps=steps,
        step_size=step_size,
        noise=0.008,
        device=device
    )

    optimizer = torch.optim.Adam(nn_energy_model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    os.makedirs("checkpoint/ebm", exist_ok=True)
    print("✅ Model and optimizer initialized — starting training...\n")

    # ------------------------------------------------------------
    # 4. Training Loop (skipped if only generating)
    # ------------------------------------------------------------
    if "--generate" not in sys.argv:
        for epoch in range(epochs):
            plt.close('all')  # ✅ clear any leftover figures from prior epochs
            print(f"\n🚀 Epoch {epoch+1}/{epochs} starting...")
            ebm.reset_metrics()
            nn_energy_model.train()

            for batch_idx, (real_imgs, _) in enumerate(train_loader):
                real_imgs = real_imgs.to(device)
                metrics = ebm.train_step(real_imgs, optimizer)

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} "
                          + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            print(f"✅ Epoch {epoch+1} metrics: "
                  + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

            # --- Plot samples safely ---
            with torch.no_grad():
                if len(ebm.buffer.examples) >= 8:
                    try:
                        plot_samples(torch.cat(ebm.buffer.examples[-8:]), n=8)
                    finally:
                        plt.close('all')  # ✅ close even if plot_samples opens figures

            # --- Validation phase ---
            ebm.reset_metrics()
            nn_energy_model.eval()
            for real_imgs, _ in test_loader:
                real_imgs = real_imgs.to(device)
                val_metrics = ebm.test_step(real_imgs)
            print("🧩 Validation: " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

        # ------------------------------------------------------------
        # 5. Save final model
        # ------------------------------------------------------------
        os.makedirs("checkpoint/ebm", exist_ok=True)
        final_path = "checkpoint/ebm/model_epoch_final.pth"
        torch.save({
            "model_state_dict": nn_energy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, final_path)

        plt.close('all')  # ✅ final cleanup
        print(f"\n✅ Training complete. Model saved to {final_path}", flush=True)

    # =============================================================
    # 6. Generate and save sample images using trained model
    # =============================================================
    import torchvision.utils as vutils

    checkpoint_path = "checkpoint/ebm/model_epoch_final.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        nn_energy_model.load_state_dict(checkpoint["model_state_dict"])
        nn_energy_model.eval()
        print(f"🔄 Loaded trained weights from {checkpoint_path}")

        # --- Generate new samples ---
        num_images = 8
        with torch.no_grad():
            if hasattr(ebm, "buffer") and len(ebm.buffer.examples) > 0:
                samples = torch.cat(ebm.buffer.examples[-num_images:])
            else:
                samples = torch.randn(num_images, 3, 32, 32, device=device)

        # --- Denormalize to [0,1] and save grid ---
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        output_dir = "checkpoint/ebm"
        os.makedirs(output_dir, exist_ok=True)
        grid_path = os.path.join(output_dir, "ebm_generated_samples.png")
        vutils.save_image(samples, grid_path, nrow=4, padding=2)
        print(f"🖼️  Sample images saved to {grid_path}")
    else:
        print("⚠️  No trained model found at checkpoint path.")
