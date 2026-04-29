"""
Inference — Stage 7.

Loads the best saved checkpoint, runs inference on each testXX_text_only.csv,
and writes testXX-pred.csv files to the predictions/ folder.

Output format matches the Gradescope autograder exactly:
    row, prediction
    0, 1
    1, 0
    ...
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import config
from model import ClinicalBERTClassifier, load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (inference — no labels)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length=config.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: Path | None = None) -> ClinicalBERTClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = checkpoint or (config.CHECKPOINTS / "best_model.pt")
    model  = ClinicalBERTClassifier(
        dropout=config.DROPOUT,
        freeze_encoder=True,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Predict] Loaded checkpoint: {ckpt}")
    return model


def predict_csv(model: ClinicalBERTClassifier, tokenizer, test_csv: Path) -> Path:
    """
    Run inference on a test CSV and write the corresponding pred CSV.

    Args:
        model:    trained ClinicalBERTClassifier
        tokenizer: matching tokenizer
        test_csv: path to testXX_text_only.csv
    Returns:
        Path to the written prediction CSV.
    """
    device = next(model.parameters()).device

    df = pd.read_csv(test_csv, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    # Support "row_id", "row", or first column as row identifier
    row_col  = df.columns[0]
    text_col = next(c for c in df.columns if "text" in c)

    rows  = df[row_col].tolist()
    texts = df[text_col].tolist()

    dataset = InferenceDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(preds)

    # Build output filename: test01_text_only.csv → test01-pred.csv
    stem     = test_csv.stem.replace("_text_only", "")
    out_path = config.PREDICTIONS / f"{stem}-pred.csv"
    pd.DataFrame({"row": rows, "prediction": predictions}).to_csv(out_path, index=False)
    print(f"[Predict] Wrote {len(predictions)} predictions → {out_path}")
    return out_path


def run_inference(checkpoint: Path | None = None):
    """Run inference on all three test CSVs."""
    config.PREDICTIONS.mkdir(exist_ok=True)
    tokenizer = load_tokenizer()
    model     = load_model(checkpoint)

    for test_csv in config.TEST_CSVS:
        if not test_csv.exists():
            print(f"[Predict] Skipping missing file: {test_csv}")
            continue
        predict_csv(model, tokenizer, test_csv)


if __name__ == "__main__":
    run_inference()
