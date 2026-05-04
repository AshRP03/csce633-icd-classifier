"""
Bio_ClinicalBERT-based binary classifier for ICD codability detection.

Architecture:
    Bio_ClinicalBERT encoder
      → mean-pool over token embeddings (masking out padding)
      → dropout
      → Linear(768 → 256) → GELU → Dropout → Linear(256 → 2)

The encoder is loaded with `local_files_only=True`. The model name resolves
through the `MODEL_PATH` environment variable so the same code works whether
the weights are at the default HuggingFace cache or at a custom local path
(needed on offline clusters).
"""
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = os.environ.get("MODEL_PATH", "emilyalsentzer/Bio_ClinicalBERT")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)


def _mean_pool(hidden_states, attention_mask):
    """Masked mean pool: average non-padding token embeddings."""
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

    def embed(self, input_ids, attention_mask):
        """
        Returns the L2-normalized pre-classifier embedding (mean-pooled BERT).
        Used by the KNN-against-gold inference path: embeddings of test
        fragments are compared via cosine similarity (= dot product when
        normalized) to embeddings of the 20 gold examples.
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _mean_pool(out.last_hidden_state, attention_mask)
        return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def unfreeze_top_layers(self, n: int = 4):
        """Unfreeze the top n encoder transformer layers + the pooler.
        Used for the second training phase, after the head has converged."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for layer in self.encoder.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True