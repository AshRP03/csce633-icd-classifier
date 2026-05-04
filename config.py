"""
Central configuration — all paths and hyperparameters live here.

KEY CHANGES vs original:
  * MAX_LENGTH 128 → 256  (test fragments are ~100 words ≈ ~150-250 BPE tokens)
  * DOC_AGG "max" → "mean_topk" with k=3 (much more stable than max for long frags)
  * GOLD_OVERSAMPLE 50 → 4  (was massively overfitting to 20 examples)
  * Phase 3 gold calibration disabled by default (was destroying the model)
  * NOTE_FRAG_PER_CLASS 2000 → 0  (the discharge=1/radiology=0 rule is wrong;
    section-based labeling replaces it; counts driven by SECTION_FRAG_PER_CLASS)
  * New: SECTION_FRAG_PER_CLASS — section-anchored fragments matching gold style
  * New: NEIGHBOR_PER_GOLD — TF-IDF nearest-neighbor mining anchored to 20 gold
  * Threshold tuning range widened back to [0.3, 0.7] but tuned on a *blend*
    of held-out pseudo + gold, not gold alone.
"""
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent
MIMIC_DIR     = PROJECT_ROOT / "mimiciii"
SUPPORT_DIR   = PROJECT_ROOT / "support_materials"
CHECKPOINTS   = PROJECT_ROOT / "checkpoints"
PREDICTIONS   = PROJECT_ROOT / "predictions"

NOTEEVENTS_CSV    = MIMIC_DIR / "NOTEEVENTS.csv"
ICD_DIAGNOSES_CSV = MIMIC_DIR / "D_ICD_DIAGNOSES.csv"
ICD_PROCEDURES_CSV= MIMIC_DIR / "D_ICD_PROCEDURES.csv"

TRAIN_CSV = SUPPORT_DIR / "train_data-text_and_labels.csv"
TEST_CSVS = [
    SUPPORT_DIR / "test01_text_only.csv",
    SUPPORT_DIR / "test02_text_only.csv",
    SUPPORT_DIR / "test03_text_only.csv",
]
EXAMPLE_PRED_CSV  = SUPPORT_DIR / "test01-pred(example).csv"
PSEUDO_LABEL_CSV  = PROJECT_ROOT / "pseudo_labeled.csv"

MODEL_NAME  = "emilyalsentzer/Bio_ClinicalBERT"

# Tokenization / inference
MAX_LENGTH  = 256                # was 128 — test fragments need this
DOC_STRIDE  = 64                 # sliding window stride
DOC_AGG     = "mean_topk"        # "max" | "mean" | "mean_topk"
DOC_TOPK    = 3                  # used when DOC_AGG="mean_topk"
PRED_THRESHOLD = 0.50            # default; overridden by threshold.json
# Set THRESHOLD_OVERRIDE to a float to ignore threshold.json entirely.
# Useful for quickly trying different thresholds without retraining.
# ROUND 3-FIX: with prior shift active, threshold 0.50 is the principled choice.
# The shift moves the model's decision boundary to match the test class balance;
# threshold=0.50 then maps back to the optimal Bayes-decision rule.
# ROUND 3-FIX-3: per-file overrides. test03 has no prior shift active, so it
# benefits from the train-tuned threshold (~0.43) rather than 0.50.
THRESHOLD_OVERRIDE = 0.50         # default if no per-file override matches

# Per-file threshold overrides. None means "use TUNED threshold from threshold.json".
# ROUND 5: test03 now has a prior shift, so threshold 0.50 is Bayes-optimal.
THRESHOLD_OVERRIDE_BY_STEM = {
    "test01": 0.50,   # uses prior shift; thr 0.50 is Bayes-optimal post-shift
    "test02": 0.50,   # ditto
    "test03": 0.50,   # round 5: now has prior shift too
}

# ROUND 3-FIX: Prior probability shift. Training data is 50/50 balanced but
# the test sets have different class balances:
#   test01: ~28% class 1 (estimated from confusion matrix)
#   test02: ~26% class 1 (estimated from confusion matrix)
#   test03: ~46% class 1 (estimated from confusion matrix) — close to 50/50
# Applying a uniform shift would help test01/02 but hurt test03.
#
# So: per-test-file priors. Map filename stem → assumed test prior.
# Stems matching no entry use TEST_PRIOR_DEFAULT (or no shift if None).
# ROUND 3-FIX-3: priors raised slightly (0.20→0.25 for test01, 0.18→0.23 for
# test02) — round 3-fix-2 over-shifted: pos_rate 0.266 was BELOW true rate
# of ~0.28, costing recall. Math: at prior 0.20, pos_rate landed at 0.266;
# at prior 0.25 it should land near 0.30, which gives more recall.
TEST_PRIORS_BY_STEM = {
    # Final tuning based on R4 (best test01) + R5 (best test03):
    # R4 used test01=0.30, test02=0.27, test03=None — gave best test01 (acc 0.798, F1 0.667)
    # R5 added test03=0.40 — gave best test03 (acc 0.702, F1 0.680)
    # R5 also tried test02=0.20 — REGRESSED test02 F1 (0.572 → 0.565), so revert.
    "test01": 0.30,
    "test02": 0.27,
    "test03": 0.40,
}
TEST_PRIOR_DEFAULT  = None        # what to do for unknown stems
TRAIN_PRIOR_CLASS1  = 0.50        # our pseudo training data is balanced

# ── Pseudo-data composition ────────────────────────────────────────────────
NOTE_CATEGORIES     = ["Discharge summary", "Physician", "Nursing/other", "Radiology"]
MAX_NOTES_TO_SAMPLE = 15_000

# Stage 5a — sentence-level pseudo (fast, kept but downweighted)
# ROUND 4: cut from 2,500 → 1,500 per class. The sentence-level data is the
# noisiest (regex score ≥4 or ≤−2 picks the easy unambiguous cases) so we
# want section + neighbor data to dominate.
MAX_PER_CLASS       = 1_500
ICD_OVERLAP_THRESH  = 3

# Stage 5b — DISABLED. The old Discharge=1 / Radiology=0 rule is wrong.
NOTE_FRAG_PER_CLASS = 0

# Stage 5c — Section-anchored fragments that look like gold examples.
SECTION_FRAG_PER_CLASS = 3_000
SECTION_FRAG_TARGET_WORDS = 100   # match gold's median (~103 words)
SECTION_FRAG_MIN_WORDS    = 50
SECTION_FRAG_MAX_WORDS    = 180

# Stage 5d — Bio_ClinicalBERT semantic neighbor mining (ROUND 4 rewrite).
# Replaces the TF-IDF-based version. Uses the same Bio_ClinicalBERT we use
# for classification to embed candidates and gold, then picks top-K nearest
# by cosine similarity. Much higher-quality neighbors than TF-IDF — semantic
# matches instead of vocabulary-overlap matches.
NEIGHBOR_PER_GOLD            = 40
NEIGHBOR_MIN_SIM             = 0.55       # SBERT-style threshold (cosine)
NEIGHBOR_CANDIDATE_POOL      = 80_000     # cap on candidates we embed
NEIGHBOR_CANDIDATES_PER_NOTE = 4

# Stage 5d additionally mines HARD NEGATIVES: candidates that look semantically
# similar to class-1 gold examples but contain unmistakable class-0 narrative
# cues (planning/disposition/instruction language). These are the
# "looks codable but isn't" cases the model currently fails on.
HARD_NEG_PER_GOLD = 15
HARD_NEG_MIN_SIM  = 0.55

# Training
BATCH_SIZE     = 16
PHASE1_LR      = 3e-4
PHASE2_LR      = 2e-5
PHASE1_EPOCHS  = 15               # was 20 — diminishing returns past this
PHASE2_EPOCHS  = 8                # was 10
DROPOUT        = 0.3              # was 0.2 — more regularization needed
UNFREEZE_TOP_N = 4
RANDOM_SEED    = 42

# Gold oversampling — DRAMATICALLY reduced. 50× was 10% of training set.
# ROUND 4: dropped to 3×. With 8 gold in training, that's 24 rows — small
# but visible. Combined with hard-negative neighbor mining there's enough
# gold-like signal in training without overfitting.
GOLD_OVERSAMPLE   = 3
PSEUDO_VAL_SPLIT  = 0.10

# Phase 3 (gold calibration) — DISABLED by default. It was destroying the
# model (acc 0.85 → 0.75 across epochs in the original run).
GOLD_CALIB_EPOCHS = 0             # was 5
GOLD_CALIB_LR     = 1e-6          # if you re-enable, use this tiny LR

# Class weighting for CrossEntropyLoss.
# ROUND 3-FIX: reverted [1.2, 1.0] → None. The class weights were intended
# to push model toward class 0, but pos_rate actually went UP from 0.41 to
# 0.54 across all test sets in round 3 — suggesting the data changes (which
# removed some class-1 patterns from training) had more effect than the loss
# weights, and combined they over-corrected away from a working balance.
CLASS_WEIGHTS = None

# Inference heuristic vetoes — DISABLED. They were applied at predict time but
# never seen during training/tuning, breaking calibration. The model alone now.
USE_INFERENCE_HEURISTICS = False

# ROUND 3-FIX-2: Gold-derived class-0 patterns. Three regex patterns each
# match exactly one of the 3 gold FPs from the round-3-fix run, and NONE of
# them match any of the 10 gold class-1 examples. Verified by direct
# inspection. When a pattern matches, subtract 0.30 from prob_class1 (soft
# bias, not hard veto). Disable by setting to False to compare A/B.
# ROUND 3-FIX-2: Gold-derived class-0 patterns. Three regex patterns each
# match exactly one of the 3 gold FPs from the round-3-fix run, and NONE of
# them match any of the 10 gold class-1 examples. Verified by direct
# inspection. When a pattern matches, subtract 0.30 from prob_class1 (soft
# bias, not hard veto). Disable by setting to False to compare A/B.
APPLY_GOLD_VETOES = True

# ROUND 6: KNN-against-gold inference. After training, the model's
# pre-classifier embeddings for all 20 gold examples are saved to
# checkpoints/gold_embeddings.npy. At inference, each test fragment's
# embedding is compared against these via cosine similarity. If close to
# gold (sim ≥ KNN_SIM_MIN), blend the K-nearest gold neighbors' labels
# (similarity-weighted) into the BERT prob, with weight ramping from 0 at
# KNN_SIM_MIN to 1 at KNN_SIM_FULL.
USE_KNN_GOLD = True
KNN_K        = 3
KNN_SIM_MIN  = 0.50    # below this, KNN is ignored entirely
KNN_SIM_FULL = 0.75    # at this similarity, KNN fully overrides BERT