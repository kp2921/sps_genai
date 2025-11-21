# app/predict_rnn.py
#
# RNN (LSTM) text generation based on Module 6: Practical - RNN
# using "The Count of Monte Cristo" from Project Gutenberg,
# with checkpoint save/load so we don't retrain every time.

import re
from collections import Counter
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#import requests
from tqdm import tqdm
from pathlib import Path

# ------------------------------------------------------
# Reproducibility
# ------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------
# Device selection (Module 6 style)
# ------------------------------------------------------
device = (
    torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(f"[RNN] Using device: {device}")

# ------------------------------------------------------
# Checkpoint configuration
# ------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoint" / "rnn"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "rnn_montecristo.pth"


# =========================
# Data loading & preprocessing
# =========================

def load_count_of_monte_cristo():
    """
    Load and preprocess 'The Count of Monte Cristo' from the LOCAL file in: sps_genai/data/montecristo.txt
    """
    # Path: app/data/montecristo.txt relative to this file
    local_path = Path(__file__).resolve().parent.parent / "data" / "montecristo.txt"
    print(f"[RNN] Loading local text from {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Keep only the main body (remove header/footer) – same logic as before
    start_idx = text.find("Chapter 1")
    end_idx = text.rfind("Chapter 5")  # same heuristic you used previously
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    # Pre-processing
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Vocabulary construction
    counter = Counter(tokens)

    # Indices 0,1 reserved for PAD and UNK
    vocab = {
        word: idx + 2
        for idx, (word, _) in enumerate(counter.most_common(9998))
    }
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    inv_vocab = {idx: word for word, idx in vocab.items()}

    # Encode tokens
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

    print(f"[RNN] Number of tokens: {len(tokens)}")
    print(f"[RNN] Vocab size: {len(vocab)}")

    return tokens, vocab, inv_vocab, encoded

# =========================
# Dataset & DataLoader
# =========================

SEQ_LEN = 30  # Module 6
BATCH_SIZE = 64


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx : idx + SEQ_LEN]),
            torch.tensor(self.data[idx + 1 : idx + SEQ_LEN + 1]),
        )


def get_dataloader(encoded):
    dataset = TextDataset(encoded)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


# =========================
# LSTM model (Module 6 arch)
# =========================

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


# =========================
# Checkpoint helpers
# =========================

def save_rnn_checkpoint(
    model: nn.Module,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    path: Path = CHECKPOINT_PATH,
):
    """Save model weights + vocab dictionaries."""
    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "inv_vocab": inv_vocab,
    }
    torch.save(ckpt, path)
    print(f"[RNN] Checkpoint saved to {path}")


def load_rnn_checkpoint(
    path: Path = CHECKPOINT_PATH,
) -> Tuple[torch.device, nn.Module, Dict[str, int], Dict[int, str]]:
    """Load model weights + vocab dictionaries from checkpoint."""
    print(f"[RNN] Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device)

    vocab = ckpt["vocab"]
    inv_vocab = ckpt["inv_vocab"]

    model = LSTMModel(vocab_size=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print("[RNN] Checkpoint loaded successfully.")
    return device, model, vocab, inv_vocab


# =========================
# Training loop (Module 6 style)
# =========================

EPOCHS = 15  # Module 6


def train_rnn_model() -> Tuple[torch.device, nn.Module, Dict[str, int], Dict[int, str]]:
    """
    End-to-end training:
      - load Monte Cristo
      - build vocab/encoded
      - build DataLoader
      - train LSTM for 15 epochs
      - save checkpoint at the end
    Returns:
      device, model, vocab, inv_vocab
    """
    tokens, vocab, inv_vocab, encoded = load_count_of_monte_cristo()
    train_loader = get_dataloader(encoded)

    model = LSTMModel(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("[RNN] Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
        for batch_number, (inputs, targets) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _ = model(inputs)
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_number % 100 == 0) or (batch_number == len(train_loader) - 1):
                train_loader_with_progress.set_postfix(
                    {"avg loss": f"{total_loss / (batch_number + 1):.4f}"}
                )

    print("[RNN] Training complete.")
    # Save checkpoint
    save_rnn_checkpoint(model, vocab, inv_vocab)
    return device, model, vocab, inv_vocab


def load_or_train_rnn_model() -> Tuple[torch.device, nn.Module, Dict[str, int], Dict[int, str]]:
    """
    Preferred entry point from main.py:

      - If a checkpoint exists, load it.
      - Otherwise, train from scratch and save checkpoint.

    Returns:
      device, model, vocab, inv_vocab
    """
    if CHECKPOINT_PATH.exists():
        try:
            return load_rnn_checkpoint(CHECKPOINT_PATH)
        except Exception as e:
            print(f"[RNN] Failed to load checkpoint ({e}); retraining from scratch...")

    # If no checkpoint or loading failed, train
    return train_rnn_model()


# =========================
# Text generation (Module 6 style)
# =========================

def generate_rnn_text(
    model: nn.Module,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    seed_text: str,
    length: int = 50,
    temperature: float = 1.0,
    dev: torch.device = None,
) -> str:
    """
    Generate text with the trained LSTM, mirroring Module 6 generate_text:
      - split seed
      - map to ids with <UNK>
      - iterative LSTM forward
      - temperature sampling with multinomial
    """
    if dev is None:
        dev = device

    model.eval()
    words = seed_text.lower().split()

    if not words:
        # if empty seed, pick random known word (not PAD/UNK) if possible
        candidates = [w for w in vocab.keys() if w not in ("<PAD>", "<UNK>")]
        if candidates:
            words = [np.random.choice(candidates)]
        else:
            words = ["<UNK>"]

    input_ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(dev)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1] / max(temperature, 1e-5)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            words.append(inv_vocab.get(next_id, "<UNK>"))
            input_ids.append(next_id)
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(dev)

    return " ".join(words)
