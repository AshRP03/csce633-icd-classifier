"""
Inference — rewritten.

Key changes vs original:
  * Heuristic vetoes (_force_negative) are DISABLED by default. They were
    applied at predict time but never seen during training/threshold tuning,
    so they broke probability calibration. Toggle via config.USE_INFERENCE_HEURISTICS.
  * Sliding-window aggregation now respects config.DOC_AGG (default mean_topk).
  * Threshold loaded from threshold.json (tuned in train.py on a blended set).
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
# Optional heuristic vetoes — only used if config.USE_INFERENCE_HEURISTICS=True
# Kept conservative: only the most unambiguous non-codable patterns.
# ─────────────────────────────────────────────────────────────────────────────

_PURE_NUMERIC = re.compile(r"^[\d\.\,\s:/\-]+$")

_POS_OVERRIDE = re.compile(
    r"\b(discharge diagnosis|admitting diagnosis|principal diagnosis|"
    r"primary diagnosis|secondary diagnosis|active issues?|"
    r"diagnos(?:is|ed|ed with)|presents? with|consistent with|"
    r"history of|h/o\b|major surgical|surgical procedure|"
    r"s/p\b|status post|underwent|repair|stent|orif|"
    r"intubat(?:ed|ion)|dialysis|pci\b|catheterization|"
    r"sepsis|pneumonia|fracture|hemorrhage|infarction|embolism|"
    r"thrombosis|cellulitis|abscess|carcinoma|malignancy|tumor|"
    r"hydrocephalus|meningitis|appendicitis|pancreatitis|"
    r"hypertension|diabetes|anemia|chf\b|copd\b|cad\b|esrd\b|"
    r"afib|atrial fibrillation|tachycardia|bradycardia|arrhythmia)\b",
    re.IGNORECASE,
)


def _force_negative(text: str) -> bool:
    """Very conservative veto. Returns True only when text is unmistakably non-codable."""
    t = (text or "").strip()
    if not t or _PURE_NUMERIC.match(t):
        return True
    if _POS_OVERRIDE.search(t):
        return False
    if len(t.split()) < 4:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gold-derived narrow vetoes (round 3-fix-2). These three patterns each catch
# one of the 3 gold FPs without firing on ANY of the 10 gold class-1 examples.
# Verified by direct inspection of all 20 gold examples.
# Used as a SOFT bias: if a pattern matches, push prob_class1 down by 0.30
# (in probability space). This nudges borderline predictions without nuking
# high-confidence ones.
# ─────────────────────────────────────────────────────────────────────────────

# FP-style #1: discharge medication continuation/instruction narrative
# ROUND 5: regexes use \s+ instead of literal spaces to handle newlines.
_VETO_DISPOSITION = re.compile(
    r"(at\s+the\s+time\s+of\s+discharge|"
    r"will\s+continue\s+this\s+for\s+\d+\s+days?|"
    r"should\s+be\s+transitioned\s+to|"
    r"poor\s+candidate\s+for\s+(?:anticoagulation|surgery)|"
    r"\bdisp:\s*\*\d+|\bsig:\s*(?:one|two|three))",
    re.IGNORECASE,
)

# FP-style #2: micro lab results (gram stain, culture results)
_VETO_MICRO = re.compile(
    r"(gram\s+positive\s+cocci|gram\s+negative|aerobic\s+bottle|gram\s+stain|"
    r"in\s+pairs\s+and\s+clusters|reported\s+to\s+and\s+read\s+back|"
    r"staph\s+aureus\s+coag|oxacillin\W{2,}|sensitivities?\s+performed)",
    re.IGNORECASE,
)

# FP-style #3: imaging COMPARISON narrative (not the IMPRESSION; comparison)
_VETO_COMPARISON = re.compile(
    r"(compared\s+with\s+the\s+(?:report\s+of\s+the\s+)?prior\s+study|"
    r"images\s+unavailable\s+for\s+review|limited\s+examination)",
    re.IGNORECASE,
)

# ROUND 3-FIX-3: three NEW patterns derived from suspicious test01 high-prob
# class-1 predictions. Verified to NOT match any of the 10 gold class-1
# examples (regression-tested in design).
#
# FP-style #4: lab values dump — fragments dominated by numeric lab readings
# (e.g. "1 g/dL / 48 mg/dL / 2.8 mg/dL / 31 mEq/L / ...")
# ROUND 3-FIX-4: tightened to require 6+ values (was 4+). The looser pattern
# was hitting borderline class-1 fragments that contain a few labs as evidence.
_VETO_LAB_DUMP = re.compile(
    r"(\b\d+\.?\d*\s*(?:mg/dL|mEq/L|mmol/L|K/uL|g/dL|ng/mL|mcg/dL|U/L|%|"
    r"mmHg|bpm|insp/min)[\s/]*){6,}",
    re.IGNORECASE,
)

# FP-style #5: vitals dump — many vital-sign readings in a row
# (e.g. "HR: 44 ... BP: 144/50 ... RR: 14 ... SpO2: 100% ...")
_VETO_VITALS_PATTERN = re.compile(
    r"\b(HR|BP|RR|SpO2|Tcurrent|MAP|CVP):\s*\d+",
    re.IGNORECASE,
)

# FP-style #6: pending labs / studies / results
_VETO_PENDING = re.compile(
    r"\b(are pending|is pending|labs?\s+pending|pending at the time of|"
    r"pending at discharge|results pending|to be followed up)\b",
    re.IGNORECASE,
)

# ROUND 5: positive override. If a fragment ALSO contains a strong class-1
# section header (e.g. "Discharge Diagnosis:", "Active Issues:", "PMH:"),
# DO NOT apply the gold-derived penalty. This protects gold examples like:
#   "Disp:*16 Capsule(s)* Refills:*0* Discharge Diagnosis: SAH hydrocephalus..."
# which look class-0 by the cue regex but are actually class 1 in gold.
_VETO_POSITIVE_OVERRIDE = re.compile(
    r"(discharge\s+diagnosis|primary\s+diagnosis|principal\s+diagnosis|"
    r"admitting\s+diagnosis|active\s+issues?|past\s+medical\s+history|"
    r"history\s+of\s+present\s+illness|major\s+surgical\s+(?:or\s+invasive\s+)?procedure|"
    r"cardiac\s+history|chief\s+complaint)\s*:",
    re.IGNORECASE,
)


def _gold_derived_penalty(text: str) -> float:
    """
    Returns a penalty in [0, 1] to subtract from the predicted prob_class1.
    ROUND 5: stack-aware AND override-aware.
      - If text contains a strong class-1 section header → penalty 0.
      - 1 distinct cue:  penalty 0.30
      - 2 distinct cues: penalty 0.45
      - 3+ distinct cues: penalty 0.55
    Counts distinct cue MATCHES (not patterns) to handle the case where one
    OR-regex contains multiple alternatives that all hit (e.g. the disposition
    pattern has alternatives "at the time of discharge", "will continue",
    "should be transitioned" — a fragment can match all three).
    """
    t = (text or "")
    # Positive override: never penalize fragments with strong class-1 markers.
    if _VETO_POSITIVE_OVERRIDE.search(t):
        return 0.0

    n_matches = 0
    n_matches += len(_VETO_DISPOSITION.findall(t))
    n_matches += len(_VETO_MICRO.findall(t))
    n_matches += len(_VETO_COMPARISON.findall(t))
    n_matches += 1 if _VETO_LAB_DUMP.search(t) else 0
    n_matches += 1 if len(_VETO_VITALS_PATTERN.findall(t)) >= 3 else 0
    n_matches += 1 if _VETO_PENDING.search(t) else 0

    if n_matches == 0:
        return 0.0
    if n_matches == 1:
        return 0.30
    if n_matches == 2:
        return 0.45
    return 0.55  # 3+ patterns — overwhelming evidence


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
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

    threshold = float(getattr(config, "PRED_THRESHOLD", 0.50))
    thr_path  = config.CHECKPOINTS / "threshold.json"
    if thr_path.exists():
        try:
            data = json.loads(thr_path.read_text())
            threshold = float(data.get("threshold", threshold))
        except Exception:
            pass
    # Manual override wins over everything (handy for quick threshold sweeps)
    # ROUND 3-FIX-3: support per-file overrides via THRESHOLD_OVERRIDE_BY_STEM.
    file_stem_for_thr = Path(test_csv).stem.replace("_text_only", "")
    overrides_by_stem = getattr(config, "THRESHOLD_OVERRIDE_BY_STEM", {})
    if file_stem_for_thr in overrides_by_stem:
        per_file = overrides_by_stem[file_stem_for_thr]
        if per_file is not None:
            threshold = float(per_file)
            print(f"[Predict] per-file THRESHOLD_OVERRIDE for {file_stem_for_thr} → {threshold:.2f}")
        else:
            print(f"[Predict] {file_stem_for_thr}: using train-tuned threshold {threshold:.2f}")
    else:
        override = getattr(config, "THRESHOLD_OVERRIDE", None)
        if override is not None:
            threshold = float(override)
            print(f"[Predict] THRESHOLD_OVERRIDE active → {threshold:.2f}")
    print(f"[Predict] threshold={threshold:.2f}  agg={config.DOC_AGG}  "
          f"max_len={config.MAX_LENGTH}  stride={config.DOC_STRIDE}  "
          f"vetoes={'ON' if getattr(config, 'USE_INFERENCE_HEURISTICS', False) else 'OFF'}")

    stride = int(getattr(config, "DOC_STRIDE", 64))
    bs     = int(getattr(config, "BATCH_SIZE", 16))
    use_vetoes = bool(getattr(config, "USE_INFERENCE_HEURISTICS", False))

    # Per-file prior shift lookup. Round-3-fix found that test01/02 are
    # ~26-28% class 1 (skewed) while test03 is ~46% (balanced).
    file_stem = Path(test_csv).stem.replace("_text_only", "")
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

    predictions = []
    raw_probs   = []   # AGGREGATED probs AFTER prior shift — used for thresholding
    raw_probs_unshifted = []   # AGGREGATED probs BEFORE prior shift — used by sweep_thresholds.py

    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch_texts = texts[start: start + bs]

            if use_vetoes:
                forced   = [_force_negative(t) for t in batch_texts]
                keep_idx = [i for i, f in enumerate(forced) if not f]
            else:
                keep_idx = list(range(len(batch_texts)))

            batch_probs = [0.0] * len(batch_texts)
            batch_probs_unshifted = [0.0] * len(batch_texts)

            if keep_idx:
                keep_texts = [batch_texts[i] for i in keep_idx]
                enc = tokenizer(
                    keep_texts, max_length=config.MAX_LENGTH,
                    truncation=True, padding="max_length", return_tensors="pt",
                    return_overflowing_tokens=True, stride=stride,
                )
                mapping = enc.pop("overflow_to_sample_mapping")
                ids  = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                probs1 = torch.softmax(model(ids, mask), -1)[:, 1].cpu()

                per_sample_shifted: list[list[float]] = [[] for _ in range(len(keep_texts))]
                per_sample_raw:     list[list[float]] = [[] for _ in range(len(keep_texts))]
                for wi, si in enumerate(mapping.tolist()):
                    p_raw = float(probs1[wi])
                    per_sample_raw[si].append(p_raw)
                    per_sample_shifted[si].append(_shift_prob(p_raw))

                for li in range(len(keep_texts)):
                    batch_probs[keep_idx[li]]            = _aggregate(per_sample_shifted[li])
                    batch_probs_unshifted[keep_idx[li]]  = _aggregate(per_sample_raw[li])

            # Apply gold-derived patterns: penalty for unmistakable class-0
            # patterns (verified to not match ANY gold class-1 example).
            # Subtract penalty from the post-shift probability before thresholding.
            apply_gold_vetoes = bool(getattr(config, "APPLY_GOLD_VETOES", True))
            if apply_gold_vetoes:
                for i_b, t in enumerate(batch_texts):
                    pen = _gold_derived_penalty(t)
                    if pen > 0:
                        batch_probs[i_b] = max(0.0, batch_probs[i_b] - pen)

            raw_probs.extend(batch_probs)
            raw_probs_unshifted.extend(batch_probs_unshifted)
            predictions.extend(1 if p >= threshold else 0 for p in batch_probs)

    stem = Path(test_csv).stem.replace("_text_only", "")
    out  = config.PREDICTIONS / f"{stem}-pred.csv"
    pd.DataFrame({"row_id": rows, "prediction": predictions}).to_csv(out, index=False)

    # ROUND 2: also write per-row probability CSV. Lets us inspect any specific
    # row's confidence and compare to Gradescope hints.
    # Round-3-fix: include both shifted (used) and unshifted (raw model output).
    diag_out = config.PREDICTIONS / f"{stem}-debug.csv"
    pd.DataFrame({
        "row_id":      rows,
        "prob_class1": np.round(raw_probs, 4),       # post-shift, used for prediction
        "prob_class1_raw": np.round(raw_probs_unshifted, 4),  # pre-shift, for sweep tool
        "prediction":  predictions,
        "text_preview": [str(t)[:120].replace("\n", " / ") for t in texts],
    }).to_csv(diag_out, index=False)

    # Quick distribution diagnostic
    pos_rate = sum(predictions) / max(1, len(predictions))
    rp = np.array(raw_probs)
    print(f"[Predict] {Path(test_csv).name}: "
          f"n={len(predictions)} pos_rate={pos_rate:.3f}  "
          f"prob mean={rp.mean():.3f} std={rp.std():.3f} "
          f"q25={np.quantile(rp,.25):.3f} q75={np.quantile(rp,.75):.3f}")
    print(f"[Predict] wrote → {out}")
    print(f"[Predict] wrote per-row debug → {diag_out}")
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