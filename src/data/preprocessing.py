"""Data preprocessing for protein secondary structure prediction.

Handles encoding of amino acid sequences and secondary structure labels,
including padding/truncation for batch processing.
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
NONSTD_AA_CHAR = "*"
PAD_TOKEN = "<PAD>"

SST8_CLASSES = list("CEHBGITS")
SST3_CLASSES = list("CEH")


class ProteinPreprocessor:
    """Encodes amino acid sequences and structure labels for model input.

    Incorporates ideas from Context.txt:
    - Handles nonstandard amino acids masked with '*' character.
    - Supports both Q8 (8-state) and Q3 (3-state) secondary structure labels.
    - Provides interface for replacing one-hot encoding with PLM embeddings
      (Input Novelty from Context.txt).

    Attributes:
        max_seq_len: Maximum sequence length (sequences are padded/truncated).
        aa_to_idx: Mapping from amino acid character to integer index.
        idx_to_aa: Reverse mapping from index to amino acid character.
        sst8_to_idx: Mapping from 8-state structure label to integer.
        sst3_to_idx: Mapping from 3-state structure label to integer.
    """

    def __init__(self, max_seq_len=512):
        self.max_seq_len = max_seq_len

        vocab = [PAD_TOKEN] + STANDARD_AMINO_ACIDS + [NONSTD_AA_CHAR]
        self.aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.vocab_size = len(vocab)

        self.sst8_to_idx = {PAD_TOKEN: 0}
        for i, s in enumerate(SST8_CLASSES, start=1):
            self.sst8_to_idx[s] = i
        self.idx_to_sst8 = {i: s for s, i in self.sst8_to_idx.items()}
        self.num_sst8_classes = len(SST8_CLASSES) + 1  # +1 for PAD

        self.sst3_to_idx = {PAD_TOKEN: 0}
        for i, s in enumerate(SST3_CLASSES, start=1):
            self.sst3_to_idx[s] = i
        self.idx_to_sst3 = {i: s for s, i in self.sst3_to_idx.items()}
        self.num_sst3_classes = len(SST3_CLASSES) + 1  # +1 for PAD

        self.pad_idx = 0

    def encode_sequence(self, seq):
        """Encode an amino acid sequence to integer indices with padding.

        Args:
            seq: String of amino acid characters.

        Returns:
            Tuple of (encoded_indices, attention_mask) as numpy arrays.
        """
        seq = seq[:self.max_seq_len]
        encoded = [self.aa_to_idx.get(aa, self.aa_to_idx[NONSTD_AA_CHAR]) for aa in seq]
        mask = [1] * len(encoded)

        pad_len = self.max_seq_len - len(encoded)
        encoded += [self.pad_idx] * pad_len
        mask += [0] * pad_len

        return np.array(encoded, dtype=np.int64), np.array(mask, dtype=np.int64)

    def encode_labels(self, label_str, label_type="sst8"):
        """Encode secondary structure label string to integer indices.

        Args:
            label_str: String of structure labels (e.g., 'CCHHHEEE').
            label_type: 'sst8' for 8-state or 'sst3' for 3-state.

        Returns:
            Numpy array of encoded label indices.
        """
        label_str = label_str[:self.max_seq_len]
        mapping = self.sst8_to_idx if label_type == "sst8" else self.sst3_to_idx
        encoded = [mapping.get(c, 0) for c in label_str]

        pad_len = self.max_seq_len - len(encoded)
        encoded += [self.pad_idx] * pad_len

        return np.array(encoded, dtype=np.int64)

    def decode_labels(self, indices, mask=None, label_type="sst8"):
        """Decode integer label indices back to structure string.

        Args:
            indices: Array of integer label indices.
            mask: Optional attention mask to determine real positions.
            label_type: 'sst8' or 'sst3'.

        Returns:
            String of decoded structure labels.
        """
        mapping = self.idx_to_sst8 if label_type == "sst8" else self.idx_to_sst3
        result = []
        for i, idx in enumerate(indices):
            if mask is not None and mask[i] == 0:
                break
            result.append(mapping.get(int(idx), "?"))
        return "".join(result)

    def decode_sequence(self, indices, mask=None):
        """Decode integer indices back to amino acid sequence.

        Args:
            indices: Array of integer amino acid indices.
            mask: Optional attention mask.

        Returns:
            String of amino acid characters.
        """
        result = []
        for i, idx in enumerate(indices):
            if mask is not None and mask[i] == 0:
                break
            result.append(self.idx_to_aa.get(int(idx), "?"))
        return "".join(result)

    def save(self, path):
        """Save preprocessor configuration to a JSON file.

        Args:
            path: File path to save the configuration.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        config = {
            "max_seq_len": self.max_seq_len,
            "aa_to_idx": self.aa_to_idx,
            "sst8_to_idx": self.sst8_to_idx,
            "sst3_to_idx": self.sst3_to_idx,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path):
        """Load preprocessor configuration from a JSON file.

        Args:
            path: File path to load the configuration from.

        Returns:
            ProteinPreprocessor instance.
        """
        with open(path, "r") as f:
            config = json.load(f)

        preprocessor = cls(max_seq_len=config["max_seq_len"])
        preprocessor.aa_to_idx = config["aa_to_idx"]
        preprocessor.idx_to_aa = {i: aa for aa, i in preprocessor.aa_to_idx.items()}
        preprocessor.sst8_to_idx = config["sst8_to_idx"]
        preprocessor.idx_to_sst8 = {int(v): k for k, v in config["sst8_to_idx"].items()}
        preprocessor.sst3_to_idx = config["sst3_to_idx"]
        preprocessor.idx_to_sst3 = {int(v): k for k, v in config["sst3_to_idx"].items()}
        preprocessor.vocab_size = len(preprocessor.aa_to_idx)
        preprocessor.num_sst8_classes = len(preprocessor.sst8_to_idx)
        preprocessor.num_sst3_classes = len(preprocessor.sst3_to_idx)
        return preprocessor
