# ============================================================
# train_diffusion.py — Diffusion Model Training Entry Script
# + Post-training image generation from best checkpoint
# ============================================================

if __name__ == "__main__":
    import torch
    from torch import optim
    from helper_lib.data_loader import get_cifar10_loader  # <-- correct helper
    from helper_lib.model import get_model
    from helper_lib.trainer import train_diffusion
    from helper_lib.utils import get_device, set_seed
    import os
    import sys

    # ------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------
    set_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    device = get_device()
    print(f"\n🧠 Environment ready — using {device.upper()}")

    # ------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------
    train_loader, val_loader = get_cifar10_loader(batch_size=64)
    print(f"✅ Loaded CIFAR-10 dataset → {len(train_loader.dataset)} train, {len(val_loader.dataset)} test samples")

    # ------------------------------------------------------------
    # 3. Initialize model, optimizer, and parameters
    # ------------------------------------------------------------
    model = get_model("diffusion")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5
    checkpoint_dir = os.path.join("checkpoint", "diffusion")

    print(f"\n🚀 Beginning diffusion training for {epochs} epochs on {device.upper()}...\n")

    # ------------------------------------------------------------
    # 4. Train the model (skipped if only generating)
    # ------------------------------------------------------------
    if "--generate" not in sys.argv:
        train_diffusion(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        print("\n✅ Diffusion training finished! Checkpoints saved in:")
        print(f"📁 {checkpoint_dir}")

    # =============================================================
    # 6. Generate and save images using the best saved diffusion model
    # =============================================================
    import torchvision.utils as vutils

    best_checkpoint = r"C:\\Users\\kp2921\\sps_genai\\checkpoint\\diffusion\\diffusion_best.pth"
    output_path = r"C:\\Users\\kp2921\\sps_genai\\checkpoint\\diffusion\\diffusion_generated_samples.png"

    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location=device)
        
        state_dict = checkpoint["model_state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("⚠️ State_dict mismatch, retrying with strict=False...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        model.to(device)
        model.eval()
        print(f"\n🔄 Loaded best checkpoint from {best_checkpoint}")

        num_images = 8
        with torch.no_grad():
            samples = model.generate(num_images=num_images, image_size=32, diffusion_steps=1000)

        samples = (samples.clamp(-1, 1) + 1) / 2.0
        vutils.save_image(samples, output_path, nrow=4, padding=2)
        print(f"🖼️  Generated samples saved to {output_path}\n")

    else:
        print(f"⚠️  No checkpoint found at {best_checkpoint}. Skipping image generation.")
