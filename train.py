"""
Training loop — Stages 6a and 6b.

Phase 1 : Frozen ClinicalBERT encoder, train classification head only.
Phase 2 : Unfreeze top N encoder layers, fine-tune at low LR.

Validation is performed on the 20 gold examples after every epoch.
Best checkpoint (highest val accuracy) is saved to checkpoints/.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

import config
from model import ClinicalBERTClassifier, load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SentenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length=config.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gold_val(tokenizer):
    """Load all 20 gold examples as a validation DataLoader."""
    df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col  = next(c for c in df.columns if "text" in c)
    label_col = next(c for c in df.columns if "label" in c)
    texts  = df[text_col].tolist()
    labels = df[label_col].astype(int).tolist()
    dataset = SentenceDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)


def evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["class0", "class1"])
    return acc, report


def run_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Training phases
# ─────────────────────────────────────────────────────────────────────────────

def train_model(pseudo_csv: str | None = None) -> ClinicalBERTClassifier:
    """
    Full two-phase training.

    Args:
        pseudo_csv: path to pseudo_labeled.csv; defaults to config.PSEUDO_LABEL_CSV
    Returns:
        The best trained model (loaded from checkpoint).
    """
    config.CHECKPOINTS.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    tokenizer = load_tokenizer()

    # Load pseudo-labeled training data
    csv_path = pseudo_csv or config.PSEUDO_LABEL_CSV
    train_df  = pd.read_csv(csv_path, dtype=str)
    train_df["label"] = train_df["label"].astype(int)
    train_dataset = SentenceDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Validation: all 20 gold examples
    val_loader = load_gold_val(tokenizer)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_ckpt = config.CHECKPOINTS / "best_model.pt"

    # ── Phase 1 : frozen encoder, train head ──────────────────────────────
    print("\n── Phase 1: Training classification head (encoder frozen) ──")
    model = ClinicalBERTClassifier(
        dropout=config.DROPOUT,
        freeze_encoder=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE1_LR,
    )

    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, report = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:02d}/{config.PHASE1_EPOCHS}  "
              f"loss={train_loss:.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint saved (val_acc={val_acc:.3f})")

    print(f"\nPhase 1 best val accuracy: {best_val_acc:.3f}")

    # ── Phase 2 : unfreeze top N encoder layers, fine-tune ────────────────
    print(f"\n── Phase 2: Fine-tuning top {config.UNFREEZE_TOP_N} encoder layers ──")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.unfreeze_top_layers(config.UNFREEZE_TOP_N)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE2_LR,
    )

    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, report = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:02d}/{config.PHASE2_EPOCHS}  "
              f"loss={train_loss:.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint saved (val_acc={val_acc:.3f})")

    print(f"\nPhase 2 best val accuracy: {best_val_acc:.3f}")
    print(report)

    # Load and return best model
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return model


if __name__ == "__main__":
    train_model()
