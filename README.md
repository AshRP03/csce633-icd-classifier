# CSCE 633 — ICD Codability Classifier

A course research project exploring text classification for clinical documentation. The model predicts whether a short clinical text fragment contains information that could support ICD coding.

> **Status:** Completed academic project and learning artifact. The repository is public for educational and portfolio purposes. It is not a clinical tool and must not be used for medical or coding decisions.

## What this project demonstrates

- Classical and neural text-classification workflow design
- Dataset construction from weak and pseudo-labeling strategies
- Domain-specific feature and hard-negative analysis
- Transformer-based representation learning
- Two-phase training and threshold-based inference
- Reproducible experimentation and model artifact management

## Approach

The project uses approximately 10,000 pseudo-labeled fragments derived from MIMIC-III clinical notes, anchored by a small set of hand-labeled examples. The data pipeline combines:

- Regex-based classification using ICD vocabulary matches
- Section-aware fragment extraction from discharge summaries and related sections
- Hard-negative mining using Bio_ClinicalBERT embeddings

The classifier uses `emilyalsెంటzer/Bio_ClinicalBERT` as a domain-specific encoder with a trainable classification head:

```text
Input text
  → truncate to 128 words
  → tokenize to at most 170 tokens
  → Bio_ClinicalBERT encoder
  → mean-pool 768-dimensional embeddings
  → dropout
  → linear 768 → 256 + GELU
  → dropout
  → linear 256 → 2
```

Training is staged:

1. Freeze the encoder and train the classification head.
2. Unfreeze selected upper encoder layers and fine-tune with a lower learning rate.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Generate data, train, and run inference
python main.py

# Reuse an existing pseudo_labeled.csv
python main.py --skip-data

# Run inference with an existing checkpoint
python main.py --predict-only
```

For offline environments, set `MODEL_PATH` to a local model directory as described in the source configuration.

## Data and responsible use

MIMIC-III data and other restricted artifacts are not included in this repository. Users must obtain access through the appropriate PhysioNet process and comply with the dataset license, credentialing, and data-use requirements.

This project is an educational experiment in applied machine learning. It does not provide medical advice, assign official ICD codes, or replace qualified clinical coding review.

## Project status

This project is no longer under active development. It remains useful as a record of my earlier work in applied NLP, classification, weak supervision, domain-specific embeddings, and model evaluation.
