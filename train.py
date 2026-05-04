"""
Training loop — rewritten.

Key changes vs original:
  * GOLD_OVERSAMPLE 50 → 4 (was 10% of training set, massive overfit)
  * Phase 3 "gold calibration" disabled by default (was destroying the model:
    in the original run, gold acc went 0.85 → 0.75 → 0.75 across epochs and
    the buggy `best_score = -1.0` init saved the worst checkpoint as "best")
  * Validation is now a blend of held-out pseudo + a held-out chunk of gold.
    We never train on the gold-validation slice, so the validation signal
    actually predicts test performance instead of over-estimating it.
  * Threshold tuning uses BOTH the pseudo-val and the held-out gold split,
    weighted toward gold (by repeating gold samples in the tuning set).
  * Tunes over [0.30, 0.70] without artificial clipping. The reason the
    original needed clipping was that Phase 3 was destroying probability
    calibration; with Phase 3 gone, full-range tuning is fine.
  * Sliding-window aggregation switched from "max" to "mean_topk" with k=3
    (configurable via config.DOC_AGG / config.DOC_TOPK).
  * Gold examples are no longer dumped into training in mass quantity. They
    serve primarily as held-out validation + threshold calibration.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

import config
from model import ClinicalBERTClassifier, load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Datasets / loaders
# ─────────────────────────────────────────────────────────────────────────────

class SentenceDataset(Dataset):
    """Pre-tokenized fixed-length dataset for fast training."""
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, max_length=config.MAX_LENGTH,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers (training-time evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_window_probs(probs: list[float]) -> float:
    if not probs:
        return 0.0
    agg = str(getattr(config, "DOC_AGG", "mean_topk")).lower()
    if agg == "mean":
        return sum(probs) / len(probs)
    if agg == "max":
        return max(probs)
    if agg == "mean_topk":
        k = max(1, int(getattr(config, "DOC_TOPK", 3)))
        top = sorted(probs, reverse=True)[:k]
        return sum(top) / len(top)
    return sum(probs) / len(probs)


def _predict_prob_class1_raw(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Raw probabilities from a pre-tokenized loader (no sliding window)."""
    model.eval()
    y_true, p1 = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            p1.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
            y_true.extend(batch["label"].tolist())
    return np.array(y_true, int), np.array(p1, float)


def _predict_prob_class1_texts(model, tokenizer, texts, device) -> np.ndarray:
    """Sliding-window inference with chosen aggregation."""
    model.eval()
    stride = int(getattr(config, "DOC_STRIDE", 64))
    bs     = int(getattr(config, "BATCH_SIZE", 16))
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch_texts = texts[start: start + bs]
            enc = tokenizer(
                batch_texts, max_length=config.MAX_LENGTH,
                truncation=True, padding="max_length", return_tensors="pt",
                return_overflowing_tokens=True, stride=stride,
            )
            mapping = enc.pop("overflow_to_sample_mapping")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            probs1 = torch.softmax(model(ids, mask), -1)[:, 1].cpu()

            per_sample: list[list[float]] = [[] for _ in range(len(batch_texts))]
            for wi, si in enumerate(mapping.tolist()):
                per_sample[si].append(float(probs1[wi]))

            for probs in per_sample:
                out.append(_aggregate_window_probs(probs))
    return np.array(out, float)


def evaluate(model, tokenizer, texts, labels, device, threshold) -> tuple[float, float, str]:
    probs1 = _predict_prob_class1_texts(model, tokenizer, texts, device)
    preds  = (probs1 >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    rep = classification_report(labels, preds, target_names=["class0", "class1"],
                                zero_division=0)
    return acc, f1, rep


# ─────────────────────────────────────────────────────────────────────────────
# Threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

def _tune_threshold(y_true: np.ndarray, p1: np.ndarray,
                    thr_min: float = 0.30, thr_max: float = 0.70,
                    n_steps: int = 81) -> tuple[float, float]:
    """Grid-search threshold maximising 0.4·acc + 0.6·F1. Returns (thr, score)."""
    best_thr, best_score = 0.5, -1.0
    for thr in np.linspace(thr_min, thr_max, n_steps):
        preds = (p1 >= thr).astype(int)
        s = 0.4 * accuracy_score(y_true, preds) \
            + 0.6 * f1_score(y_true, preds, zero_division=0)
        if s > best_score:
            best_score, best_thr = s, float(thr)
    return best_thr, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def _load_gold():
    df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    tc = next(c for c in df.columns if "text" in c)
    lc = next(c for c in df.columns if "label" in c)
    df = df[[tc, lc]].rename(columns={tc: "text", lc: "label"})
    df["label"] = df["label"].astype(int)
    return df


def train_model(pseudo_csv=None):
    config.CHECKPOINTS.mkdir(exist_ok=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    tokenizer = load_tokenizer()

    # ── Load data ─────────────────────────────────────────────────────────────
    gold_df = _load_gold()
    gold_text_set = set(gold_df["text"].tolist())

    full_df = pd.read_csv(pseudo_csv or str(config.PSEUDO_LABEL_CSV), dtype=str)
    full_df["label"] = full_df["label"].astype(int)
    pseudo_df = full_df[~full_df["text"].isin(gold_text_set)].reset_index(drop=True)
    pseudo_df = pseudo_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    pseudo_df = pseudo_df.sample(frac=1, random_state=config.RANDOM_SEED) \
                          .reset_index(drop=True)
    print(f"[Train] Pseudo (non-gold, deduped): {len(pseudo_df):,} rows")
    print(f"[Train] Pseudo class balance — "
          f"class0: {(pseudo_df['label']==0).sum():,} "
          f"class1: {(pseudo_df['label']==1).sum():,}")

    # ── Splits ────────────────────────────────────────────────────────────────
    # Pseudo split
    n_val_pseudo = max(200, int(len(pseudo_df) * float(getattr(config, "PSEUDO_VAL_SPLIT", 0.10))))
    pseudo_val   = pseudo_df.iloc[:n_val_pseudo].reset_index(drop=True)
    pseudo_train = pseudo_df.iloc[n_val_pseudo:].reset_index(drop=True)

    # ROUND 4: stratified 8/12 split. The previous all-20-as-val approach
    # gave good validation signal but starved the model of direct gold
    # exposure. With our new BERT-mined neighbors providing more gold-like
    # data already in training, adding a small amount of actual gold
    # (8 examples × 3 oversample = 24 rows) should help direct calibration
    # without dominating the loss like the original 50× oversampling did.
    rng = np.random.RandomState(config.RANDOM_SEED)
    gold_idx_by_class = {c: list(gold_df.index[gold_df["label"]==c]) for c in (0, 1)}
    for c in gold_idx_by_class:
        rng.shuffle(gold_idx_by_class[c])

    # 4 of each class → 8 in training; 6 of each → 12 in validation
    gold_val_idx, gold_tr_idx = [], []
    for c, idxs in gold_idx_by_class.items():
        gold_tr_idx.extend(idxs[:4])
        gold_val_idx.extend(idxs[4:])

    gold_val   = gold_df.loc[gold_val_idx].reset_index(drop=True)
    gold_train = gold_df.loc[gold_tr_idx].reset_index(drop=True)
    print(f"[Train] Gold split (round-4 stratified): "
          f"train={len(gold_train)}, val={len(gold_val)}")

    # Mild gold oversampling — 3× of 8 = 24 rows in training set
    gold_oversample = int(getattr(config, "GOLD_OVERSAMPLE", 3))
    gold_repeated = pd.concat([gold_train] * gold_oversample, ignore_index=True)

    train_df = pd.concat([pseudo_train, gold_repeated], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=config.RANDOM_SEED) \
                       .reset_index(drop=True)

    print(f"[Train] Training set: {len(train_df):,} "
          f"({len(pseudo_train):,} pseudo + {len(gold_repeated):,} gold "
          f"[{len(gold_train)}×{gold_oversample}])")
    print(f"[Train] Pseudo-val: {len(pseudo_val):,} | Gold-val: {len(gold_val):,}")
    print(f"[Train] Train class balance — "
          f"class0: {(train_df['label']==0).sum():,} "
          f"class1: {(train_df['label']==1).sum():,}")

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        SentenceDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer),
        batch_size=config.BATCH_SIZE, shuffle=True)
    pseudo_val_loader = DataLoader(
        SentenceDataset(pseudo_val["text"].tolist(), pseudo_val["label"].tolist(), tokenizer),
        batch_size=config.BATCH_SIZE, shuffle=False)

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw = getattr(config, "CLASS_WEIGHTS", None)
    if cw is not None:
        weight = torch.tensor(cw, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
        print(f"[Train] Class weights: {cw}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_ckpt = config.CHECKPOINTS / "best_model.pt"
    PATIENCE  = 4

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: head-only
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Phase 1: head only (encoder frozen) ──")
    model = ClinicalBERTClassifier(dropout=config.DROPOUT, freeze_encoder=True).to(device)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE1_LR, weight_decay=0.01)

    best_score, no_imp = -1.0, 0
    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        loss = run_epoch(model, train_loader, optim, criterion, device)

        # Pseudo-val signal (cheap)
        y_pv, p1_pv = _predict_prob_class1_raw(model, pseudo_val_loader, device)
        pv_preds = (p1_pv >= 0.5).astype(int)
        pv_acc = accuracy_score(y_pv, pv_preds)
        pv_f1  = f1_score(y_pv, pv_preds, zero_division=0)

        # Gold-val signal (the one that matters; small but expensive due to sliding window)
        gv_acc, gv_f1, _ = evaluate(model, tokenizer, gold_val["text"].tolist(),
                                     gold_val["label"].tolist(), device, 0.5)

        # Combined score: 70% gold-val + 30% pseudo-val
        score = 0.7 * (0.4*gv_acc + 0.6*gv_f1) + 0.3 * (0.4*pv_acc + 0.6*pv_f1)

        print(f"  E{epoch:02d} loss={loss:.4f}  pv_acc={pv_acc:.3f} pv_f1={pv_f1:.3f}  "
              f"gv_acc={gv_acc:.3f} gv_f1={gv_f1:.3f}  score={score:.3f}")
        if score > best_score:
            best_score, no_imp = score, 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"    ✓ saved (best={best_score:.3f})")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop at epoch {epoch}"); break

    print(f"Phase 1 best score: {best_score:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: unfreeze top layers
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n── Phase 2: fine-tune top {config.UNFREEZE_TOP_N} encoder layers ──")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.unfreeze_top_layers(config.UNFREEZE_TOP_N)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.PHASE2_LR, weight_decay=0.05)

    no_imp = 0
    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        loss = run_epoch(model, train_loader, optim, criterion, device)
        if loss < 0.05:
            print(f"  loss too low ({loss:.4f}), stopping"); break

        y_pv, p1_pv = _predict_prob_class1_raw(model, pseudo_val_loader, device)
        pv_preds = (p1_pv >= 0.5).astype(int)
        pv_acc = accuracy_score(y_pv, pv_preds)
        pv_f1  = f1_score(y_pv, pv_preds, zero_division=0)

        gv_acc, gv_f1, _ = evaluate(model, tokenizer, gold_val["text"].tolist(),
                                     gold_val["label"].tolist(), device, 0.5)
        score = 0.7 * (0.4*gv_acc + 0.6*gv_f1) + 0.3 * (0.4*pv_acc + 0.6*pv_f1)

        print(f"  E{epoch:02d} loss={loss:.4f}  pv_acc={pv_acc:.3f} pv_f1={pv_f1:.3f}  "
              f"gv_acc={gv_acc:.3f} gv_f1={gv_f1:.3f}  score={score:.3f}")
        if score > best_score:
            best_score, no_imp = score, 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"    ✓ saved (best={best_score:.3f})")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop at epoch {epoch}"); break

    print(f"Phase 2 best score: {best_score:.3f}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Diagnostic eval on FULL gold set (both train and val portions)
    full_gold_acc, full_gold_f1, full_gold_rep = evaluate(
        model, tokenizer, gold_df["text"].tolist(),
        gold_df["label"].tolist(), device, 0.5)
    print(f"\n[Diag] Full gold @ 0.5: acc={full_gold_acc:.3f} f1={full_gold_f1:.3f}")
    print(full_gold_rep)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 (optional): gold calibration — DISABLED by default.
    # If you re-enable, the implementation below avoids the original bugs:
    #   * best_score initialised properly
    #   * tiny LR (1e-6 by default)
    #   * very few epochs
    #   * tracks held-out gold_val (not training gold)
    # ─────────────────────────────────────────────────────────────────────────
    calibrate_epochs = int(getattr(config, "GOLD_CALIB_EPOCHS", 0))
    if calibrate_epochs > 0 and len(gold_train) > 0:
        print(f"\n── Phase 3: gold calibration ({calibrate_epochs} epochs) ──")
        gold_loader = DataLoader(
            SentenceDataset(gold_train["text"].tolist() * 5,
                            gold_train["label"].tolist() * 5, tokenizer),
            batch_size=4, shuffle=True)

        model.unfreeze_top_layers(config.UNFREEZE_TOP_N)
        for p in model.classifier.parameters():
            p.requires_grad = True

        calib_optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(getattr(config, "GOLD_CALIB_LR", 1e-6)),
            weight_decay=0.01)

        # Initialise best with current model's gold-val score (proper init!)
        gv_acc0, gv_f10, _ = evaluate(model, tokenizer, gold_val["text"].tolist(),
                                       gold_val["label"].tolist(), device, 0.5)
        best_gv = 0.4*gv_acc0 + 0.6*gv_f10
        no_imp = 0
        print(f"  baseline gv: acc={gv_acc0:.3f} f1={gv_f10:.3f} score={best_gv:.3f}")

        for epoch in range(1, calibrate_epochs + 1):
            loss = run_epoch(model, gold_loader, calib_optim, criterion, device)
            gv_acc, gv_f1, _ = evaluate(model, tokenizer, gold_val["text"].tolist(),
                                         gold_val["label"].tolist(), device, 0.5)
            s = 0.4*gv_acc + 0.6*gv_f1
            print(f"  E{epoch:02d} loss={loss:.4f} gv_acc={gv_acc:.3f} "
                  f"gv_f1={gv_f1:.3f} score={s:.3f}")
            if s > best_gv:
                best_gv, no_imp = s, 0
                torch.save(model.state_dict(), best_ckpt)
                print("    ✓ saved")
            else:
                no_imp += 1
                if no_imp >= 2:
                    print("    early stop"); break

        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold tuning — on a BLEND of held-out pseudo + held-out gold
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Threshold tuning (blended pseudo-val + gold-val) ──")

    # Get probs on pseudo-val (sliding window for fair comparison with test)
    p1_pv = _predict_prob_class1_texts(model, tokenizer,
                                        pseudo_val["text"].tolist(), device)
    y_pv  = np.array(pseudo_val["label"].tolist(), int)

    # Get probs on gold-val (now the FULL 20 gold)
    p1_gv = _predict_prob_class1_texts(model, tokenizer,
                                        gold_val["text"].tolist(), device)
    y_gv  = np.array(gold_val["label"].tolist(), int)

    # ROUND 2: weight gold harder. Gold is the only data point that actually
    # resembles the test distribution; pseudo is structurally biased toward
    # whatever the section regex caught. Weight gold so it counts ~50% in the
    # blend (was 25%).
    gold_weight = max(1, len(p1_pv) // (2 * max(1, len(p1_gv))))
    p1_blend = np.concatenate([p1_pv] + [p1_gv] * gold_weight)
    y_blend  = np.concatenate([y_pv]  + [y_gv]  * gold_weight)

    thr, sc = _tune_threshold(y_blend, p1_blend, thr_min=0.30, thr_max=0.70)

    # Also compute the pseudo-only and gold-only optima for visibility
    thr_p, sc_p = _tune_threshold(y_pv, p1_pv, thr_min=0.30, thr_max=0.70)
    thr_g, sc_g = _tune_threshold(y_gv, p1_gv, thr_min=0.30, thr_max=0.70)

    print(f"  pseudo-val opt: thr={thr_p:.2f} score={sc_p:.3f}  (n={len(y_pv)})")
    print(f"  gold-val   opt: thr={thr_g:.2f} score={sc_g:.3f}  (n={len(y_gv)})")
    print(f"  BLEND      opt: thr={thr:.2f} score={sc:.3f}  "
          f"(gold_weight={gold_weight})")

    # Sanity: if gold-val threshold differs wildly from pseudo, lean toward gold.
    # (Round 1 used average; round 2 leans gold because we now have full 20.)
    if abs(thr_g - thr_p) > 0.15:
        thr = round((2 * thr_g + thr_p) / 3, 2)   # 2/3 gold + 1/3 pseudo
        print(f"  (gold and pseudo diverge → leaning toward gold: thr={thr:.2f})")

    thr_path = config.CHECKPOINTS / "threshold.json"
    with open(thr_path, "w") as f:
        json.dump({"threshold": thr,
                   "thr_pseudo_only": thr_p,
                   "thr_gold_only":   thr_g,
                   "thr_blend":       thr},
                   f, indent=2)
    print(f"[Train] Tuned threshold {thr:.2f} → {thr_path}")

    # ── DIAGNOSTIC: per-row gold predictions ──────────────────────────────
    # Always written. Lets us see exactly which gold examples the model is
    # failing on, even without a separate predict-side debug step.
    diag_path = config.CHECKPOINTS / "gold_diagnostic.csv"
    diag_df = pd.DataFrame({
        "text":      gold_val["text"].tolist(),
        "true_label":  y_gv,
        "prob_class1": p1_gv.round(4),
        "pred_at_thr": (p1_gv >= thr).astype(int),
        "correct":     ((p1_gv >= thr).astype(int) == y_gv).astype(int),
    })
    # Truncate text in CSV for readability
    diag_df["text"] = diag_df["text"].str.slice(0, 200) + "..."
    diag_df.to_csv(diag_path, index=False)
    print(f"[Train] Wrote per-row gold diagnostic → {diag_path}")
    print(f"[Train] Gold mistakes by class: "
          f"FN (pred 0, true 1): {((diag_df['pred_at_thr']==0) & (diag_df['true_label']==1)).sum()}, "
          f"FP (pred 1, true 0): {((diag_df['pred_at_thr']==1) & (diag_df['true_label']==0)).sum()}")
    print(f"[Train] Tuned threshold {thr:.2f} → {thr_path}")

    g_acc, g_f1, _ = evaluate(model, tokenizer, gold_df["text"].tolist(),
                               gold_df["label"].tolist(), device, thr)
    print(f"[Train] Full gold @ tuned thr: acc={g_acc:.3f} f1={g_f1:.3f}")

    return model


if __name__ == "__main__":
    train_model()