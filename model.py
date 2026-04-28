"""
ClinicalBERT binary classifier for ICD codability detection.

Architecture:
    Tokenizer → ClinicalBERT encoder (frozen) → [CLS] embedding (768-dim)
              → Dropout(0.3) → Linear(768 → 2) → CrossEntropyLoss
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 128


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        return self.classifier(self.dropout(cls_embedding))  # [batch, 2]

    def unfreeze_top_layers(self, n: int = 2):
        """Unfreeze the top n transformer encoder layers for fine-tuning phase 2."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_layers = self.encoder.encoder.layer
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
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
