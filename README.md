# CSCE 633 - ICD Codability Classifier

Binary text classifier that predicts whether a clinical text fragment (up to **128 words**) is "ICD-codable"—i.e., whether it contains medical information that an ICD coder could use to assign a diagnosis or procedure code.

## Project Overview

The classifier is trained on **~10k pseudo-labeled fragments** from MIMIC-III clinical notes, anchored on 20 hand-labeled gold examples. The data pipeline uses multiple labeling strategies:
- **Regex-based sentence classification** on ICD vocabulary matches
- **Section-anchored fragments** from discharge summaries (Discharge Diagnosis, PMH, etc.)
- **Hard negative mining** via Bio_ClinicalBERT embeddings (fragments that look codable but contain disposition language)

## Model Architecture

Bio_ClinicalBERT encoder (`emilyalsentzer/Bio_ClinicalBERT`) with a trainable classification head:

```
Input → Truncate to 128 words → Tokenize (max 170 tokens)
      → Bio_ClinicalBERT encoder (frozen initially)
      → Mean-pool embeddings (768-dim)
      → Dropout(0.3)
      → Linear(768 → 256) → GELU → Dropout
      → Linear(256 → 2) → CrossEntropyLoss
```

**Two-phase training:**
1. Freeze encoder, train classification head only
2. Unfreeze top encoder layers + fine-tune at lower learning rate

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py                # Full pipeline: data generation → training → inference
python main.py --skip-data    # Reuse existing pseudo_labeled.csv (skip data pipeline)
python main.py --predict-only # Inference only (requires saved checkpoint)
```

## Notes

- Max input: 128 words (truncated if longer)
- Fully offline inference; uses `MODEL_PATH` environment variable for custom model paths (required for offline clusters like Grace)
- MIMIC-III data is excluded from this repo per course policy
- Outputs: best model weights, decision threshold, and gold example embeddings for KNN-against-gold inference
