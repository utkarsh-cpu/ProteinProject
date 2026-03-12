"""Transformer-based model for protein secondary structure prediction.

Architectural Novelty from Context.txt:
Uses self-attention (Transformer encoder) to capture long-range dependencies
between amino acids. While CNNs find local patterns (e.g., local helices),
the self-attention mechanism weighs the importance of distant residues against
each other, capturing global folding structure across the entire sequence.

Input Novelty from Context.txt:
The model uses learnable amino acid embeddings by default but is designed to
accept pre-computed embeddings from Protein Language Models (PLMs) like ESM-2
or ProtBERT through the `external_embeddings` parameter. PLM embeddings encode
evolutionary and chemical context learned from millions of protein sequences.

Output Novelty from Context.txt:
The model supports an optional CRF (Conditional Random Field) layer that
replaces standard Softmax for sequence labeling. The CRF captures transition
probabilities between secondary structure states, preventing physically
impossible predictions.
"""

import math
import logging

import torch
import torch.nn as nn

from src.model.crf import CRFLayer
from src.model.config import ModelConfig

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ProteinTransformer(nn.Module):
    """Transformer encoder model for protein secondary structure prediction.

    Predicts both 8-state (Q8) and 3-state (Q3) secondary structure from
    amino acid sequences using multi-head self-attention.

    Args:
        config: ModelConfig instance with model hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        self.sst8_head = nn.Linear(config.d_model, config.num_sst8_classes)
        self.sst3_head = nn.Linear(config.d_model, config.num_sst3_classes)

        self.use_crf = config.use_crf
        if self.use_crf:
            self.crf_sst8 = CRFLayer(config.num_sst8_classes, pad_idx=config.pad_idx)
            self.crf_sst3 = CRFLayer(config.num_sst3_classes, pad_idx=config.pad_idx)

        self._init_weights()
        logger.info(
            "ProteinTransformer: d_model=%d, heads=%d, layers=%d, CRF=%s",
            config.d_model, config.nhead, config.num_encoder_layers, config.use_crf,
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform, preserving CRF constraints."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "crf" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask, sst8_labels=None, sst3_labels=None,
                external_embeddings=None):
        """Forward pass for training or inference.

        Args:
            input_ids: (batch, seq_len) amino acid token indices.
            attention_mask: (batch, seq_len) binary mask (1=real, 0=pad).
            sst8_labels: Optional (batch, seq_len) 8-state labels for loss computation.
            sst3_labels: Optional (batch, seq_len) 3-state labels for loss computation.
            external_embeddings: Optional (batch, seq_len, d_model) pre-computed PLM
                                 embeddings to replace learned embeddings.

        Returns:
            dict with keys:
                'loss': Combined loss (if labels provided).
                'sst8_logits': (batch, seq_len, num_sst8_classes) emission scores.
                'sst3_logits': (batch, seq_len, num_sst3_classes) emission scores.
                'sst8_preds': Predicted 8-state labels.
                'sst3_preds': Predicted 3-state labels.
        """
        if external_embeddings is not None:
            x = external_embeddings
        else:
            x = self.embedding(input_ids)

        x = self.pos_encoding(x)

        src_key_padding_mask = attention_mask == 0
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        sst8_logits = self.sst8_head(x)
        sst3_logits = self.sst3_head(x)

        result = {"sst8_logits": sst8_logits, "sst3_logits": sst3_logits}

        if sst8_labels is not None and sst3_labels is not None:
            if self.use_crf:
                loss_sst8 = self.crf_sst8(sst8_logits, sst8_labels, attention_mask)
                loss_sst3 = self.crf_sst3(sst3_logits, sst3_labels, attention_mask)
            else:
                ce_sst8 = nn.CrossEntropyLoss(ignore_index=self.config.pad_idx)
                ce_sst3 = nn.CrossEntropyLoss(ignore_index=self.config.pad_idx)
                loss_sst8 = ce_sst8(sst8_logits.view(-1, self.config.num_sst8_classes),
                                     sst8_labels.view(-1))
                loss_sst3 = ce_sst3(sst3_logits.view(-1, self.config.num_sst3_classes),
                                     sst3_labels.view(-1))
            result["loss"] = loss_sst8 + loss_sst3
            result["loss_sst8"] = loss_sst8
            result["loss_sst3"] = loss_sst3

        if self.use_crf:
            result["sst8_preds"] = self.crf_sst8.decode(sst8_logits, attention_mask)
            result["sst3_preds"] = self.crf_sst3.decode(sst3_logits, attention_mask)
        else:
            result["sst8_preds"] = sst8_logits.argmax(dim=-1)
            result["sst3_preds"] = sst3_logits.argmax(dim=-1)

        return result
