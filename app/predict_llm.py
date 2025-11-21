# app/predict_llm.py
#
# Fine-tune GPT-2 (openai-community/gpt2) on SQuAD (Module 9 style)
# and then apply a simple RL-style post-training step (Module 11 style)
# to encourage answers in a specific format:
#
#   "That is a great question ... let me know if you have any other questions"

from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

device = (
    torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

print(f"[LLM] Using device: {device}")

# ------------------------------------------------------
# Checkpoints: <project_root>/checkpoint/llm 
# ------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoint" / "llm"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# Hyperparameters 
# ------------------------------------------------------
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 5
LR = 5e-5
RL_STEPS = 20    # small RL post-training loop

# Fix seeds (helps reproducibility)
torch.manual_seed(42)

# =========================
# FORMAT REQUIREMENT
# =========================
FORMAT_PREFIX = "That is a great question! "
FORMAT_SUFFIX = ". Let me know if you have any other questions."


# =========================
# 1. DATA: SQuAD QA Dataset
# =========================

class SquadQADataset(Dataset):
    """
    Simple QA dataset for causal LM fine-tuning:
      text = "Question: <question> Answer: <answer>"
    We train GPT-2 as a causal LM.
    NOTE: We do NOT pad here; the collator handles padding per batch.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = MAX_LENGTH,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = load_dataset("rajpurkar/squad", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example["question"]
        # Just take the first answer if there are multiple
        answer = example["answers"]["text"][0] if example["answers"]["text"] else ""

        text = f"Question: {question} Answer: {answer}"

        # We tokenize WITHOUT padding, truncation only.
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension and return dict; collator will pad & create labels.
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def create_dataloaders(tokenizer):
    """
    Create train & validation DataLoaders using SQuAD and a
    DataCollatorForLanguageModeling that handles padding and labels.
    """
    train_dataset = SquadQADataset(
        tokenizer=tokenizer,
        split="train",
        max_length=MAX_LENGTH,
        max_samples=2000,   # subset to avoid huge training time
    )
    val_dataset = SquadQADataset(
        tokenizer=tokenizer,
        split="validation",
        max_length=MAX_LENGTH,
        max_samples=500,
    )

    # Data collator for causal LM (no MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is a causal LM
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,   # dynamic padding + labels
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    return train_loader, val_loader


# =========================
# 2. BASE MODEL: GPT-2
# =========================

def init_base_llm():
    """
    Initialize GPT-2 base model and tokenizer.
    """
    print("[LLM] Initializing base GPT-2 and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # GPT-2 does not define a pad_token by default; use eos_token as pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return model, tokenizer


# =========================
# 3. SUPERVISED FINE-TUNING (Module 9-style)
# =========================

def train_llm_model() -> Tuple[torch.device, AutoModelForCausalLM, AutoTokenizer]:
    """
    Fine-tune GPT-2 on SQuAD subset (Module 9 GPT activity),
    using a collator for dynamic padding & labels.
    """
    model, tokenizer = init_base_llm()
    train_loader, _ = create_dataloaders(tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("[LLM] Starting supervised fine-tuning on SQuAD subset...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        progress = tqdm(
            iterable=train_loader,
            ncols=120,
            desc=f"[LLM] Epoch {epoch+1}/{EPOCHS}",
        )

        for batch_idx, batch in enumerate(progress):
            # batch = {"input_ids", "attention_mask", "labels"} from collator
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress.set_postfix({"loss": f"{avg_loss:.4f}"})

    print("[LLM] Supervised fine-tuning complete.")

    # Save checkpoint (Hugging Face format)
    model.save_pretrained(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    print(f"[LLM] Saved fine-tuned model & tokenizer to: {CHECKPOINT_DIR}")

    # OPTIONAL: RL post-training for format
    rl_post_train_format_enforcement(model, tokenizer, steps=RL_STEPS)

    # Save again after RL (if you want the final format-enforced model)
    model.save_pretrained(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    print(f"[LLM] Saved RL-post-trained model & tokenizer to: {CHECKPOINT_DIR}")

    return device, model, tokenizer


def load_or_train_llm_model() -> Tuple[torch.device, AutoModelForCausalLM, AutoTokenizer]:
    """
    Entry point for main.py:
        - If checkpoint exists at <root>/checkpoint/llm, load it.
        - Otherwise, fine-tune from GPT-2 base and save checkpoint.
    """
    config_path = CHECKPOINT_DIR / "config.json"
    if config_path.exists():
        print(f"[LLM] Loading fine-tuned model from {CHECKPOINT_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(CHECKPOINT_DIR))
        model.to(device)
        print("[LLM] Loaded fine-tuned GPT-2.")
        return device, model, tokenizer

    # No checkpoint yet → train
    return train_llm_model()


# =========================
# 4. RL POST-TRAINING (Module 11-style)
# =========================

def compute_format_reward(output_text: str) -> float:
    """
    Simple reward function:
      +1 if it starts with FORMAT_PREFIX
      +1 if it ends with FORMAT_SUFFIX
    """
    text = output_text.strip()
    reward = 0.0
    if text.startswith(FORMAT_PREFIX):
        reward += 1.0
    if text.endswith(FORMAT_SUFFIX):
        reward += 1.0
    return reward


def rl_post_train_format_enforcement(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    steps: int = RL_STEPS,
):
    """
    Tiny RL-ish loop to nudge model toward the desired format.
    This is a simplified REINFORCE-style update:

      - Sample a response
      - Compute reward based on format
      - Compute loss = -reward * (negative log-likelihood)
      - Backpropagate
    """
    print(f"[LLM] Starting RL-style post-training for format ({steps} steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    example_prompt = "Question: What is machine learning? Answer:"

    for step in range(steps):
        # 1) Generate a candidate answer
        generated = generate_llm_text(
            model=model,
            tokenizer=tokenizer,
            prompt=example_prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            dev=device,
            raw_only=True,  # get raw text for reward
        )

        reward = compute_format_reward(generated)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

        # 2) Recompute loss on the same prompt (teacher forcing)
        inputs = tokenizer(
            example_prompt,
            return_tensors="pt",
        ).to(device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )
        # outputs.loss is average NLL; we scale by -reward
        loss = outputs.loss * (-reward_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[LLM][RL] step {step+1}/{steps} - reward={reward:.2f}")

    print("[LLM] RL-style post-training complete.")


# =========================
# 5. GENERATION HELPER
# =========================

def generate_llm_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    dev: Optional[torch.device] = None,
    raw_only: bool = False,
) -> str:
    """
    Generate text with the fine-tuned (and optionally RL post-trained) GPT-2.
    If raw_only=True, return only the raw decoded string.
    Otherwise, FastAPI layer will wrap it with the required prefix/suffix.
    """
    if dev is None:
        dev = device

    model.eval()

    if not prompt or prompt.strip() == "":
        prompt = "Question: What is reinforcement learning? Answer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(dev)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if raw_only:
        return generated_text

    return generated_text
