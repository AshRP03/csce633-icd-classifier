"""
Central configuration — all paths and hyperparameters live here.
Edit MIMIC_DIR and SUPPORT_DIR if your folder layout differs.
"""

from pathlib import Path

# ── Directory roots ────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
MIMIC_DIR      = PROJECT_ROOT / "mimiciii"
SUPPORT_DIR    = PROJECT_ROOT / "support_materials"
CHECKPOINTS    = PROJECT_ROOT / "checkpoints"
PREDICTIONS    = PROJECT_ROOT / "predictions"

# ── MIMIC-III source files ─────────────────────────────────────────────────────
NOTEEVENTS_CSV      = MIMIC_DIR / "NOTEEVENTS.csv"
ICD_DIAGNOSES_CSV   = MIMIC_DIR / "D_ICD_DIAGNOSES.csv"
ICD_PROCEDURES_CSV  = MIMIC_DIR / "D_ICD_PROCEDURES.csv"

# ── Instructor-provided files ──────────────────────────────────────────────────
TRAIN_CSV           = SUPPORT_DIR / "train_data-text_and_labels.csv"
TEST_CSVS           = [
    SUPPORT_DIR / "test01_text_only.csv",
    SUPPORT_DIR / "test02_text_only.csv",
    SUPPORT_DIR / "test03_text_only.csv",
]
EXAMPLE_PRED_CSV    = SUPPORT_DIR / "test01-pred(example).csv"

# ── Pseudo-label output (written by data_pipeline, read by train) ──────────────
PSEUDO_LABEL_CSV    = PROJECT_ROOT / "pseudo_labeled.csv"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME          = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH          = 128

# For inference on long note fragments: tokenize with overlapping windows
# and aggregate per-row probabilities.
DOC_STRIDE          = 32        # overlap between windows (tokens)
DOC_AGG             = "max"    # {"max", "mean"} - mean avoids single noisy window spikes
PRED_THRESHOLD      = 0.40      # Sweet spot between internal/external

# ── Data pipeline ─────────────────────────────────────────────────────────────
NOTE_CATEGORIES     = ["Discharge summary", "Physician", "Nursing/other", "Radiology"]
MAX_NOTES_TO_SAMPLE = 15_000     # rows from NOTEEVENTS before sentence splitting
MAX_PER_CLASS       = 4_000     # pseudo-label cap per class after labeling
ICD_OVERLAP_THRESH  = 3         # Balance: more diverse training data

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE          = 16        # larger batch for stable gradients
PHASE1_LR           = 1e-4      # head-only training - standard rate
PHASE2_LR           = 2e-5      # Conservative fine-tuning
PHASE1_EPOCHS       = 15        # Full head training - do NOT reduce
PHASE2_EPOCHS       = 10         # Short fine-tuning - do NOT increase
DROPOUT             = 0.3       # Moderate regularization
UNFREEZE_TOP_N      = 4         # Conservative encoder adaptation - do NOT increase
# VAL_SPLIT           = 0.1       # use all 20 gold examples as validation (not train)
RANDOM_SEED         = 42

EXCLUDE_GOLD_FROM_TRAIN = True

# final calibration on the 20 gold-labeled examples.
GOLD_CALIB_EPOCHS  = 0
GOLD_CALIB_REPEAT  = 3    # just 3x oversample, not memorizing
GOLD_CALIB_LR      = 5e-6