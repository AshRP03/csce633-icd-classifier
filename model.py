"""
ClinicalBERT binary classifier for ICD codability detection.

Architecture:
    Tokenizer → ClinicalBERT encoder → mean-pooled embedding (768)
              → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→2)

Same architecture as the original — the issues were in DATA, not architecture.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.environ.get("MODEL_PATH", "emilyalsentzer/Bio_ClinicalBERT")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)


def _mean_pool(hidden_states, attention_mask):
    """Masked mean pooling over token embeddings."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _mean_pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

    def unfreeze_top_layers(self, n: int = 4):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for layer in self.encoder.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True


if __name__ == "__main__":
    import config
    tokenizer = load_tokenizer()
    model = ClinicalBERTClassifier(dropout=config.DROPOUT, freeze_encoder=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")