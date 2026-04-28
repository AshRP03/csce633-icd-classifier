# CSCE 633 — ICD Codability Classifier

Binary sentence classifier that predicts whether a sentence from a clinical note contains ICD-codable medical information.

## Architecture

ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) pretrained encoder with a trainable classification head.

```
Tokenizer → ClinicalBERT encoder (frozen) → [CLS] embedding (768-dim)
          → Dropout(0.3) → Linear(768 → 2) → CrossEntropyLoss
```

Training is two-phase:
1. Train classification head only (encoder frozen)
2. Unfreeze top 2 encoder layers and fine-tune at low LR

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python model.py   # downloads ClinicalBERT and runs a sanity check forward pass
```

## Notes

- Max input length: 128 tokens (per project spec)
- Fully offline inference — no external APIs
- MIMIC-III data is excluded from this repo per course policy
