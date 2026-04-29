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

# ── Data pipeline ─────────────────────────────────────────────────────────────
NOTE_CATEGORIES     = ["Discharge summary", "Physician", "Nursing/other", "Radiology"]
MAX_NOTES_TO_SAMPLE = 8_000     # rows from NOTEEVENTS before sentence splitting
MAX_PER_CLASS       = 4_000     # pseudo-label cap per class after labeling
ICD_OVERLAP_THRESH  = 2         # min ICD vocab term hits to call class 1

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE          = 16
PHASE1_LR           = 1e-3      # head-only training
PHASE2_LR           = 1e-5      # fine-tuning top encoder layers
PHASE1_EPOCHS       = 10
PHASE2_EPOCHS       = 2
DROPOUT             = 0.3
UNFREEZE_TOP_N      = 2         # number of encoder layers to unfreeze in phase 2
VAL_SPLIT           = 0.0       # use all 20 gold examples as validation (not train)
RANDOM_SEED         = 42
