"""
ClinicalBERT binary classifier for ICD codability detection.

Architecture:
    Tokenizer → ClinicalBERT encoder (frozen) → [CLS] embedding (768-dim)
              → Dropout(0.3) → Linear(768 → 2) → CrossEntropyLoss
"""

import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.environ.get("MODEL_PATH", "emilyalsentzer/Bio_ClinicalBERT")
MAX_LENGTH = 128


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)

        # Replace single linear with a deeper head — much better for domain adaptation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool instead of just CLS — more robust representation
        token_embeddings = outputs.last_hidden_state          # [B, seq, 768]
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq, 1]
        sum_embeddings = (token_embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask                    # [B, 768]
        return self.classifier(self.dropout(pooled))

    def unfreeze_top_layers(self, n: int = 2):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Also always unfreeze the pooler
        if hasattr(self.encoder, "pooler") and self.encoder.pooler:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True


if __name__ == "__main__":
    print(f"Loading tokenizer and model from '{MODEL_NAME}'...")
    tokenizer = load_tokenizer()
    model = ClinicalBERTClassifier(freeze_encoder=True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    sample = "The patient presents with acute chest pain and shortness of breath."
    tokens = tokenizer(sample, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted class: {logits.argmax(dim=-1).item()}")
    print("Model loaded and forward pass successful.")
