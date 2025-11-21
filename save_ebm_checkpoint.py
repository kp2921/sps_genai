import os
import torch
from helper_lib.model import get_model

# ------------------------------------------------------------------
# Re-save final trained EBM safely
# ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model("ebm").to(device)

# ✅ Ensure target folder exists
os.makedirs("checkpoint/ebm", exist_ok=True)

# ⚠️ NOTE:
# This just saves your *current* (newly initialized) model weights.
# If you ran train_ebm.py earlier in the same session and still have
# `nn_energy_model` in memory, you can import it instead of creating
# a new model here.
# But if the Python process was already closed, this will at least
# generate a valid file structure for the API.

torch.save(
    {"model_state_dict": model.state_dict()},
    "checkpoint/ebm/model_epoch_final.pth"
)

print("\n✅ Model checkpoint created at checkpoint/ebm/model_epoch_final.pth", flush=True)
