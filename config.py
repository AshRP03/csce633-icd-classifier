"""
Central configuration for the ICD codability classifier.

This file holds all paths and hyperparameters. Touching anything here is enough
to change pipeline / training / inference behaviour without editing code.

The classifier predicts whether a clinical text fragment is "codable" — i.e.
whether an ICD coder could use the text to assign a diagnosis code. The
pipeline trains on pseudo-labelled MIMIC-III fragments anchored on 20
hand-labelled gold examples.
"""
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
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
PSEUDO_LABEL_CSV = PROJECT_ROOT / "pseudo_labeled.csv"

# ─── Model / tokenization ───────────────────────────────────────────────────
MODEL_NAME  = "emilyalsentzer/Bio_ClinicalBERT"
MAX_WORDS   = 128         # max input words (per project spec)
MAX_LENGTH  = 256              # max tokens after BERT tokenization
DOC_STRIDE  = 64               # sliding-window stride for long fragments
DOC_AGG     = "mean_topk"      # aggregation across windows: max | mean | mean_topk
DOC_TOPK    = 3                # top-k window probabilities to average
PRED_THRESHOLD = 0.50          # default; per-file overrides below take precedence

# ─── Data pipeline ──────────────────────────────────────────────────────────
NOTE_CATEGORIES     = ["Discharge summary", "Physician", "Nursing/other", "Radiology"]
MAX_NOTES_TO_SAMPLE = 15_000   # for the sentence-level pseudo stage (5a)

# Stage 5a: sentence-level pseudo-labels via a hand-crafted regex scorer.
# Kept small because section + neighbour data is more reliable.
MAX_PER_CLASS       = 1_500
ICD_OVERLAP_THRESH  = 3

# Stage 5c: section-anchored fragments. Fragments are pulled from named
# sections inside discharge summaries (Discharge Diagnosis, PMH, etc.) and
# labelled by section type. Imaging IMPRESSION sections are content-classified.
SECTION_FRAG_PER_CLASS    = 3_000
SECTION_FRAG_TARGET_WORDS = 100   # matches gold examples' median (~103 words)
SECTION_FRAG_MIN_WORDS    = 50
SECTION_FRAG_MAX_WORDS    = 180

# Stage 5d: neighbour mining. For each gold example, find the K most-similar
# fragments in MIMIC using Bio_ClinicalBERT semantic embeddings and label them
# the same as the gold example. Also mines HARD NEGATIVES — fragments
# semantically close to class-1 gold but containing class-0 narrative cues
# (planning/disposition/instruction language). Hard negatives teach the model
# the difference between "evidences a diagnosis" and "plans treatment forward".
NEIGHBOR_PER_GOLD            = 40
NEIGHBOR_MIN_SIM             = 0.55
NEIGHBOR_CANDIDATE_POOL      = 80_000   # cap to keep BERT embedding under budget
NEIGHBOR_CANDIDATES_PER_NOTE = 4
HARD_NEG_PER_GOLD = 15
HARD_NEG_MIN_SIM  = 0.55

# ─── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE     = 16
PHASE1_LR      = 3e-4           # head-only training
PHASE2_LR      = 2e-5           # top encoder layers + head
PHASE1_EPOCHS  = 15
PHASE2_EPOCHS  = 8
DROPOUT        = 0.3
UNFREEZE_TOP_N = 4              # how many top encoder layers to unfreeze in Phase 2
RANDOM_SEED    = 42

# Gold examples are split: a few are oversampled into training for direct
# concept exposure; the rest are held out for validation and threshold tuning.
GOLD_OVERSAMPLE   = 3
PSEUDO_VAL_SPLIT  = 0.10

CLASS_WEIGHTS = None            # uniform; loss is CrossEntropy with label smoothing

# ─── Inference ──────────────────────────────────────────────────────────────
# Prior probability shift: training data is 50/50 balanced, but the test sets
# have different class-1 rates. Per-file priors correct for this so threshold
# 0.50 becomes the Bayes-optimal decision rule for each file.
TEST_PRIORS_BY_STEM = {
    "test01": 0.30,
    "test02": 0.27,
    "test03": 0.40,
}
TEST_PRIOR_DEFAULT = None
TRAIN_PRIOR_CLASS1 = 0.50

# Per-file threshold overrides. With prior shift active, 0.50 is correct.
THRESHOLD_OVERRIDE_BY_STEM = {
    "test01": 0.50,
    "test02": 0.50,
    "test03": 0.50,
}
THRESHOLD_OVERRIDE = 0.50

# Gold-derived inference-time penalty. A small set of regex patterns matches
# unmistakable class-0 narratives (medication instructions, micro lab dumps,
# imaging comparison narrative, etc.). When a pattern matches, subtract 0.30
# from the predicted prob. Each pattern was verified against all 20 gold
# examples and never matches a class-1 gold (with positive-override safety).
APPLY_GOLD_VETOES = True

# KNN-against-gold inference. After training, the trained model's embeddings
# of all 20 gold examples are saved. At inference, each test fragment's
# embedding is compared against these via cosine similarity. If close enough,
# the K nearest gold neighbours' labels are blended into the prediction —
# weighted by similarity, with the blend strength ramping from 0 (at SIM_MIN)
# to 1 (at SIM_FULL). This lets gold ground-truth directly correct BERT
# classifier mistakes when the test fragment is recognisably gold-like.
USE_KNN_GOLD = True
KNN_K        = 3
KNN_SIM_MIN  = 0.50
KNN_SIM_FULL = 0.75