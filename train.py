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
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json

import config
from model import ClinicalBERTClassifier, load_tokenizer
from predict import _force_negative


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
    """Load all gold examples as a validation DataLoader."""
    df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col  = next(c for c in df.columns if "text" in c)
    label_col = next(c for c in df.columns if "label" in c)
    texts  = df[text_col].tolist()
    labels = df[label_col].astype(int).tolist()
    dataset = SentenceDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)


def load_gold_val_examples() -> tuple[list[str], list[int]]:
    """Load gold texts/labels for evaluation & threshold tuning."""
    df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = next(c for c in df.columns if "text" in c)
    label_col = next(c for c in df.columns if "label" in c)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def _load_gold_text_set() -> set[str]:
    """Return the set of gold-labeled sentence texts (for leak-free training)."""
    df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = next(c for c in df.columns if "text" in c)
    return set(df[text_col].astype(str).tolist())


def _predict_prob_class1_texts(
    model: ClinicalBERTClassifier,
    tokenizer,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    """Predict P(class=1) for each text using the same inference logic as Stage 7."""
    model.eval()

    stride = int(getattr(config, "DOC_STRIDE", 32))
    agg = str(getattr(config, "DOC_AGG", "max")).lower()
    bs = int(getattr(config, "BATCH_SIZE", 16))

    probs_all: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch_texts = texts[start : start + bs]

            forced = [_force_negative(t) for t in batch_texts]
            keep_local_idx = [i for i, f in enumerate(forced) if not f]

            batch_probs = [0.0] * len(batch_texts)

            if keep_local_idx:
                keep_texts = [batch_texts[i] for i in keep_local_idx]

                enc = tokenizer(
                    keep_texts,
                    max_length=config.MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_overflowing_tokens=True,
                    stride=stride,
                )
                mapping = enc.pop("overflow_to_sample_mapping")

                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                logits = model(ids, mask)
                probs1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()

                per_sample: list[list[float]] = [[] for _ in range(len(keep_texts))]
                for win_idx, sample_idx in enumerate(mapping.tolist()):
                    per_sample[sample_idx].append(float(probs1[win_idx]))

                for local_i, probs in enumerate(per_sample):
                    if not probs:
                        p = 0.0
                    elif agg == "mean":
                        p = float(sum(probs) / len(probs))
                    else:
                        p = float(max(probs))
                    batch_probs[keep_local_idx[local_i]] = p

            probs_all.extend(batch_probs)

    return np.asarray(probs_all, dtype=float)


def evaluate(
    model: ClinicalBERTClassifier,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    threshold: float,
) -> tuple[float, float, str]:
    probs1 = _predict_prob_class1_texts(model, tokenizer, texts, device)
    preds = (probs1 >= float(threshold)).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    report = classification_report(labels, preds, target_names=["class0", "class1"])
    return acc, f1, report


def _predict_prob_class1(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, p_class1) arrays."""
    model.eval()
    y_true: list[int] = []
    p1: list[float] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            p1.extend(probs1.tolist())
            y_true.extend(batch["label"].numpy().tolist())
    return np.asarray(y_true, dtype=int), np.asarray(p1, dtype=float)


def _tune_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Choose a threshold that maximizes the same weighted score used in training."""
    best_thr = 0.5
    best_score = -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        preds = (p1 >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        score = 0.4 * acc + 0.6 * f1
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


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

    gold_texts, gold_labels = load_gold_val_examples()

    # Load pseudo-labeled training data
    csv_path = pseudo_csv or config.PSEUDO_LABEL_CSV
    train_df  = pd.read_csv(csv_path, dtype=str)
    train_df["label"] = train_df["label"].astype(int)

    # IMPORTANT: Prevent data leakage.
    # data_pipeline.py appends the 20 gold-labeled examples into pseudo_labeled.csv,
    # but we also use those same 20 examples for validation. Exclude them from the
    # training split so val_acc is meaningful.
    exclude_gold = getattr(config, "EXCLUDE_GOLD_FROM_TRAIN", True)
    if exclude_gold and "text" in train_df.columns:
        gold_text_set = _load_gold_text_set()
        before = len(train_df)
        train_df = train_df[~train_df["text"].isin(gold_text_set)].reset_index(drop=True)
        removed = before - len(train_df)
        if removed:
            print(f"[Train] Excluded {removed} gold validation rows from training")

    if len(train_df) == 0:
        raise ValueError(
            "Training set is empty after excluding gold validation examples. "
            "Set config.EXCLUDE_GOLD_FROM_TRAIN = False to disable this behavior."
        )
    train_dataset = SentenceDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Validation: all gold examples (note fragments)
    val_loader = load_gold_val(tokenizer)

    # Compute class weights to handle potential imbalance in pseudo-labeled data
    # labels_array = np.array(train_df["label"].tolist())
    # class_counts = np.bincount(labels_array)
    # class_weights = torch.tensor(
    #     [1.0 / (cnt + 1e-6) for cnt in class_counts],  # inverse frequency weighting
    #     dtype=torch.float32,
    #     device=device
    # # )
    # class_weights = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device)
    # # class_weights = class_weights / class_weights.sum() * 2  # normalize to ~1 on average
    # print(f"[Train] Class weights (for imbalance): {class_weights.cpu().tolist()}")
    
    # criterion = nn.CrossEntropyLoss(
    #     weight=class_weights,
    #     label_smoothing=0.1
    # )
    criterion = FocalLoss(
        alpha=torch.tensor([2.0, 1.0], dtype=torch.float32, device=device),
        gamma=2.0
    )
    best_score = 0.0
    PATIENCE = 5  # stop if no improvement for 5 epochs
    no_improve = 0
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

        threshold = 0.50
        val_acc, val_f1, report = evaluate(
            model, tokenizer, gold_texts, gold_labels, device, threshold
        )
        print(f"  Epoch {epoch:02d}/{config.PHASE1_EPOCHS}  "
          f"loss={train_loss:.4f}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  ")
        
        current_score = 0.4 * val_acc + 0.6 * val_f1  # weighted score prioritizing F1
        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint (val_acc={val_acc:.3f}, val_f1={val_f1:.3f}, score={current_score:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nPhase 1 best score: {best_score:.3f}")

    # ── Phase 2 : unfreeze top N encoder layers, fine-tune ────────────────
    best_score = 0.0
    no_improve = 0

    print(f"\n── Phase 2: Fine-tuning top {config.UNFREEZE_TOP_N} encoder layers ──")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.unfreeze_top_layers(config.UNFREEZE_TOP_N)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE2_LR,
        weight_decay=0.05,   # stronger regularization
    )

    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        threshold = 0.50
        val_acc, val_f1, report = evaluate(
            model, tokenizer, gold_texts, gold_labels, device, threshold
        )

        # Stop if train loss is suspiciously low — model is memorizing
        if train_loss < 0.05:
            print(f"  Train loss too low ({train_loss:.4f}) — stopping to prevent memorization")
            break
        
        print(f"  Epoch {epoch:02d}/{config.PHASE2_EPOCHS}  "
          f"loss={train_loss:.4f}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  ")

        current_score = 0.4 * val_acc + 0.6 * val_f1
        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best checkpoint (val_acc={val_acc:.3f}, val_f1={val_f1:.3f}, score={current_score:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nPhase 2 best score: {best_score:.3f}")
    print(report)

    # Load and return best model
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # short calibration fine-tune on the 20 gold examples.
    # This often improves F1 on held-out evaluation because gold labels are higher quality.
    calibrate_epochs = int(getattr(config, "GOLD_CALIB_EPOCHS", 0))

    if calibrate_epochs > 0:
        # Phase 3: Fine-tune on gold labels
        print("\n-- Phase 3: Gold label fine-tuning (calibration) ──")
        gold_df = pd.read_csv(config.TRAIN_CSV, dtype=str)
        gold_df.columns = [c.strip().lower() for c in gold_df.columns]
        text_col  = next(c for c in gold_df.columns if "text" in c)
        label_col = next(c for c in gold_df.columns if "label" in c)

        # Oversample gold 5x to give it enough weight vs pseudo data
        gold_texts  = gold_df[text_col].tolist() * 5
        gold_labels = gold_df[label_col].astype(int).tolist() * 5

        gold_dataset = SentenceDataset(gold_texts, gold_labels, tokenizer)
        gold_loader  = DataLoader(gold_dataset, batch_size=8, shuffle=True)

        model.unfreeze_top_layers(config.UNFREEZE_TOP_N)
        for p in model.classifier.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-6,
            weight_decay=0.01,
        )

        best_gold_score = -1.0
        no_improve_gold = 0
        for epoch in range(1, calibrate_epochs + 1):
            train_loss = run_epoch(model, gold_loader, optimizer, criterion, device)
            threshold = 0.50
            val_acc, val_f1, _ = evaluate(
                model, tokenizer, gold_texts, gold_labels, device, threshold
            )
            current_score = 0.4 * val_acc + 0.6 * val_f1
            print(f"  Gold Epoch {epoch:02d}  loss={train_loss:.4f}  "
                f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")
            if current_score > best_gold_score:
                best_gold_score = current_score
                no_improve_gold = 0
                torch.save(model.state_dict(), best_ckpt)
                print("  ✓ Saved gold checkpoint")
            else:
                no_improve_gold += 1
                if no_improve_gold >= 2:
                    print("  Early stopping gold phase")
                    break

        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Auto-tune threshold on gold validation examples and save it for inference.
    # This matters because we use sliding-window + aggregation in Stage 7.
    p1 = _predict_prob_class1_texts(model, tokenizer, gold_texts, device)
    y_true = np.asarray(gold_labels, dtype=int)
    threshold = _tune_threshold(y_true, p1)
    thr_path = config.CHECKPOINTS / "threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f)
    print(f"[Train] Tuned threshold {threshold:.2f} → {thr_path}")

    return model


if __name__ == "__main__":
    train_model()
