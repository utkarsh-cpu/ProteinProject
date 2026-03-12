"""Model configuration for the Protein Transformer."""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class ModelConfig:
    """Configuration for the ProteinTransformer model.

    Attributes:
        vocab_size: Number of amino acid tokens (including PAD and special tokens).
        num_sst8_classes: Number of 8-state secondary structure classes (including PAD).
        num_sst3_classes: Number of 3-state secondary structure classes (including PAD).
        max_seq_len: Maximum sequence length.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
        use_crf: Whether to use CRF layer for output (Output Novelty from Context.txt).
        pad_idx: Padding index for input tokens and labels.
    """

    vocab_size: int = 22
    num_sst8_classes: int = 9
    num_sst3_classes: int = 4
    max_seq_len: int = 512
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    use_crf: bool = True
    pad_idx: int = 0

    def save(self, path):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
