"""
Inference: produce per-row class predictions for each test CSV.

The prediction for one fragment is a four-step pipeline:

  1. BERT classifier produces per-window class-1 probabilities (sliding window
     over long fragments) and they are aggregated via mean-of-top-k.

  2. Per-file prior shift: the training data is balanced 50/50 but each test
     file has its own class-1 prevalence. Shifting the model's logits by
     log(p_test/(1-p_test)) - log(p_train/(1-p_train)) makes threshold 0.50
     the Bayes-optimal decision for each file.

  3. Gold-derived penalty: a small set of regex patterns matches unmistakable
     class-0 narratives (medication instructions, micro lab dumps, imaging
     comparison narrative, etc.). Each pattern was verified against all 20
     gold examples and never fires on a class-1 gold (with a positive override
     for fragments containing strong class-1 section headers). When a pattern
     matches, subtract 0.30 from the predicted probability.

  4. KNN-against-gold blend: each test fragment's BERT embedding is compared
     to the embeddings of all 20 gold examples (saved by train.py). If close
     to gold (cosine sim ≥ KNN_SIM_MIN), blend the K nearest gold neighbours'
     labels into the prediction with weight ramping from 0 (at SIM_MIN) to 1
     (at SIM_FULL). This injects ground-truth gold information directly at
     inference time and is the largest single contributor to test accuracy.

Final prediction = (post-shift, post-penalty, post-KNN-blend prob) ≥ threshold.
"""

import re
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

import config
from model import ClinicalBERTClassifier, load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Gold-derived class-0 patterns
# ─────────────────────────────────────────────────────────────────────────────
#
# Each of these regexes captures an unambiguous class-0 narrative pattern.
# All were checked against the 20 gold examples and verified NEVER to match
# a class-1 gold (with the positive-override below as a safety net).
#
# When a pattern matches, the predicted prob_class1 is reduced by 0.30 — small
# enough that genuinely high-confidence class-1 predictions still survive,
# large enough to flip borderline cases from class-1 to class-0.

# Discharge medication continuation / forward-looking dispositional language
_VETO_DISPOSITION = re.compile(
    r"(at\s+the\s+time\s+of\s+discharge|"
    r"will\s+continue\s+this\s+for\s+\d+\s+days?|"
    r"should\s+be\s+transitioned\s+to|"
    r"poor\s+candidate\s+for\s+(?:anticoagulation|surgery)|"
    r"\bdisp:\s*\*\d+|\bsig:\s*(?:one|two|three))",
    re.IGNORECASE,
)

# Microbiology lab result dumps
_VETO_MICRO = re.compile(
    r"(gram\s+positive\s+cocci|gram\s+negative|aerobic\s+bottle|gram\s+stain|"
    r"in\s+pairs\s+and\s+clusters|reported\s+to\s+and\s+read\s+back|"
    r"staph\s+aureus\s+coag|oxacillin\W{2,}|sensitivities?\s+performed)",
    re.IGNORECASE,
)

# Imaging comparison narrative (NOT the IMPRESSION — just describing change)
_VETO_COMPARISON = re.compile(
    r"(compared\s+with\s+the\s+(?:report\s+of\s+the\s+)?prior\s+study|"
    r"images\s+unavailable\s+for\s+review|limited\s+examination)",
    re.IGNORECASE,
)

# Pure numeric lab-value dumps (e.g. "1 g/dL / 48 mg/dL / 2.8 mg/dL ...")
_VETO_LAB_DUMP = re.compile(
    r"(\b\d+\.?\d*\s*(?:mg/dL|mEq/L|mmol/L|K/uL|g/dL|ng/mL|mcg/dL|U/L|%|"
    r"mmHg|bpm|insp/min)[\s/]*){6,}",
    re.IGNORECASE,
)

# Vitals readout patterns (HR/BP/RR/SpO2 with values)
_VETO_VITALS_PATTERN = re.compile(
    r"\b(HR|BP|RR|SpO2|Tcurrent|MAP|CVP):\s*\d+",
    re.IGNORECASE,
)

# "Pending" labs / studies — purely forward-looking, not codable
_VETO_PENDING = re.compile(
    r"\b(are pending|is pending|labs?\s+pending|pending at the time of|"
    r"pending at discharge|results pending|to be followed up)\b",
    re.IGNORECASE,
)

# Positive override: if the fragment contains a strong class-1 section header,
# DO NOT apply any of the above penalties. Two of the 20 gold class-1 examples
# contain "Sig: One" and "Disp:*16" but follow them with a Discharge Diagnosis
# listing — those need to stay class 1.
_VETO_POSITIVE_OVERRIDE = re.compile(
    r"(discharge\s+diagnosis|primary\s+diagnosis|principal\s+diagnosis|"
    r"admitting\s+diagnosis|active\s+issues?|past\s+medical\s+history|"
    r"history\s+of\s+present\s+illness|major\s+surgical\s+(?:or\s+invasive\s+)?procedure|"
    r"cardiac\s+history|chief\s+complaint)\s*:",
    re.IGNORECASE,
)


def _gold_derived_penalty(text: str) -> float:
    """Return a probability subtractor in [0, 0.30] for unmistakable class-0 text."""
    t = (text or "")
    if _VETO_POSITIVE_OVERRIDE.search(t):
        return 0.0
    if (_VETO_DISPOSITION.search(t) or _VETO_MICRO.search(t)
            or _VETO_COMPARISON.search(t) or _VETO_LAB_DUMP.search(t)
            or len(_VETO_VITALS_PATTERN.findall(t)) >= 3
            or _VETO_PENDING.search(t)):
        return 0.30
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Word truncation (enforce MAX_WORDS limit)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_max_words(text: str, max_words: int = None) -> str:
    """Truncate text to max_words. Defaults to config.MAX_WORDS if not specified."""
    if max_words is None:
        max_words = int(getattr(config, "MAX_WORDS", 128))
    if not text or max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ─────────────────────────────────────────────────────────────────────────────
# Window-probability aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(probs: list[float]) -> float:
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


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = checkpoint or (config.CHECKPOINTS / "best_model.pt")
    model  = ClinicalBERTClassifier(dropout=config.DROPOUT, freeze_encoder=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    print(f"[Predict] Loaded checkpoint: {ckpt}")
    return model


def predict_csv(model, tokenizer, test_csv):
    device = next(model.parameters()).device
    df = pd.read_csv(test_csv, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    row_col  = ("row_id" if "row_id" in df.columns
                else ("row" if "row" in df.columns else df.columns[0]))
    text_col = next(c for c in df.columns if "text" in c)
    rows  = df[row_col].tolist()
    texts = df[text_col].astype(str).tolist()
    file_stem = Path(test_csv).stem.replace("_text_only", "")

    # ── Resolve threshold (per-file overrides take precedence) ──────────────
    threshold = float(getattr(config, "PRED_THRESHOLD", 0.50))
    thr_path  = config.CHECKPOINTS / "threshold.json"
    if thr_path.exists():
        try:
            data = json.loads(thr_path.read_text())
            threshold = float(data.get("threshold", threshold))
        except Exception:
            pass
    overrides_by_stem = getattr(config, "THRESHOLD_OVERRIDE_BY_STEM", {})
    if file_stem in overrides_by_stem and overrides_by_stem[file_stem] is not None:
        threshold = float(overrides_by_stem[file_stem])
    elif getattr(config, "THRESHOLD_OVERRIDE", None) is not None:
        threshold = float(config.THRESHOLD_OVERRIDE)
    print(f"[Predict] threshold={threshold:.2f}")

    # ── Resolve per-file prior shift ────────────────────────────────────────
    priors_by_stem = getattr(config, "TEST_PRIORS_BY_STEM", {})
    test_prior = priors_by_stem.get(file_stem,
                                     getattr(config, "TEST_PRIOR_DEFAULT", None))
    train_prior = getattr(config, "TRAIN_PRIOR_CLASS1", 0.5)
    prior_logit_shift = 0.0
    if test_prior is not None and 0 < test_prior < 1 and 0 < train_prior < 1:
        import math
        prior_logit_shift = (math.log(test_prior / (1 - test_prior))
                              - math.log(train_prior / (1 - train_prior)))
        print(f"[Predict]  prior shift for {file_stem}: train={train_prior:.2f} "
              f"test={test_prior:.2f} → logit shift {prior_logit_shift:+.3f}")
    else:
        print(f"[Predict]  no prior shift for {file_stem}")

    def _shift_prob(p: float) -> float:
        if prior_logit_shift == 0.0:
            return p
        eps = 1e-7
        p = max(eps, min(1 - eps, p))
        import math
        logit = math.log(p / (1 - p))
        new_logit = logit + prior_logit_shift
        return 1.0 / (1.0 + math.exp(-new_logit))

    # ── Set up KNN-against-gold (if enabled and embeddings exist) ───────────
    use_knn = bool(getattr(config, "USE_KNN_GOLD", False))
    gold_embeds = None
    gold_labels = None
    if use_knn:
        ge_path = config.CHECKPOINTS / "gold_embeddings.npy"
        gl_path = config.CHECKPOINTS / "gold_labels.npy"
        if ge_path.exists() and gl_path.exists():
            gold_embeds = np.load(ge_path)         # (20, 768) L2-normalized
            gold_labels = np.load(gl_path)         # (20,) int 0/1
        else:
            print("[Predict] KNN requested but no gold embeddings found — disabling")
            use_knn = False

    knn_K        = int(getattr(config, "KNN_K", 3))
    knn_sim_min  = float(getattr(config, "KNN_SIM_MIN", 0.50))
    knn_sim_full = float(getattr(config, "KNN_SIM_FULL", 0.75))

    def _knn_prob_and_alpha(test_emb: np.ndarray) -> tuple[float, float]:
        """Look up K nearest gold neighbours for a test embedding.
        Returns (knn_prob, alpha) where alpha is the blend weight."""
        sims = gold_embeds @ test_emb         # cosine sim (everything is normalized)
        order = np.argsort(-sims)
        topk_idx = order[:knn_K]
        topk_sims = sims[topk_idx]
        topk_labels = gold_labels[topk_idx]
        max_sim = float(topk_sims.max())
        if max_sim < knn_sim_min:
            return 0.5, 0.0  # no useful neighbour, let BERT decide alone
        w = np.clip(topk_sims, 0, None)
        if w.sum() < 1e-9:
            return 0.5, 0.0
        knn_prob = float((w * topk_labels).sum() / w.sum())
        # Linear ramp: alpha=0 at SIM_MIN, alpha=1 at SIM_FULL
        alpha = (max_sim - knn_sim_min) / max(1e-9, knn_sim_full - knn_sim_min)
        alpha = float(max(0.0, min(1.0, alpha)))
        return knn_prob, alpha

    # ── Main inference loop ─────────────────────────────────────────────────
    stride = int(getattr(config, "DOC_STRIDE", 64))
    bs     = int(getattr(config, "BATCH_SIZE", 16))
    apply_gold_vetoes = bool(getattr(config, "APPLY_GOLD_VETOES", True))

    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch_texts = texts[start: start + bs]
            # Enforce MAX_WORDS limit before any processing
            batch_texts = [_truncate_to_max_words(t) for t in batch_texts]
            batch_probs = [0.0] * len(batch_texts)

            # Step 1: BERT classifier with sliding-window aggregation
            enc = tokenizer(
                batch_texts, max_length=config.MAX_LENGTH,
                truncation=True, padding="max_length", return_tensors="pt",
                return_overflowing_tokens=True, stride=stride,
            )
            mapping = enc.pop("overflow_to_sample_mapping")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            probs1 = torch.softmax(model(ids, mask), -1)[:, 1].cpu()

            # Group window-probs by source sample, then prior-shift each window
            # individually before aggregating. (Equivalent to shifting log-odds,
            # which is what the prior-correction math intends.)
            per_sample: list[list[float]] = [[] for _ in range(len(batch_texts))]
            for wi, si in enumerate(mapping.tolist()):
                per_sample[si].append(_shift_prob(float(probs1[wi])))
            for li in range(len(batch_texts)):
                batch_probs[li] = _aggregate(per_sample[li])

            # Step 2: gold-derived penalty (already conservative and overridden
            # by class-1 section headers when applicable)
            if apply_gold_vetoes:
                for i_b, t in enumerate(batch_texts):
                    pen = _gold_derived_penalty(t)
                    if pen > 0:
                        batch_probs[i_b] = max(0.0, batch_probs[i_b] - pen)

            # Step 3: KNN-against-gold blend. Single-pass embedding (no sliding
            # window) for consistency with how gold was embedded in train.py.
            if use_knn:
                enc_emb = tokenizer(
                    batch_texts, max_length=config.MAX_LENGTH,
                    truncation=True, padding="max_length", return_tensors="pt",
                )
                e_ids  = enc_emb["input_ids"].to(device)
                e_mask = enc_emb["attention_mask"].to(device)
                test_embs = model.embed(e_ids, e_mask).cpu().numpy()  # (B, 768)

                for i_b in range(len(batch_texts)):
                    knn_prob, alpha = _knn_prob_and_alpha(test_embs[i_b])
                    if alpha > 0:
                        blended = (1.0 - alpha) * batch_probs[i_b] + alpha * knn_prob
                        batch_probs[i_b] = max(0.0, min(1.0, blended))

            predictions.extend(1 if p >= threshold else 0 for p in batch_probs)

    out = config.PREDICTIONS / f"{file_stem}-pred.csv"
    pd.DataFrame({"row_id": rows, "prediction": predictions}).to_csv(out, index=False)
    pos_rate = sum(predictions) / max(1, len(predictions))
    print(f"[Predict] {Path(test_csv).name}: n={len(predictions)} "
          f"pos_rate={pos_rate:.3f} → {out}")
    return out


def run_inference(checkpoint=None):
    config.PREDICTIONS.mkdir(exist_ok=True)
    tokenizer = load_tokenizer()
    model     = load_model(checkpoint)
    for test_csv in config.TEST_CSVS:
        if not Path(test_csv).exists():
            print(f"[Predict] skipping missing: {test_csv}")
            continue
        predict_csv(model, tokenizer, test_csv)


if __name__ == "__main__":
    run_inference()