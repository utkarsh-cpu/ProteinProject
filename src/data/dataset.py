"""PyTorch Dataset for protein secondary structure data."""

import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import ProteinPreprocessor

logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein sequences and secondary structure labels.

    Args:
        dataframe: pandas DataFrame with 'seq', 'sst8', and 'sst3' columns.
        preprocessor: ProteinPreprocessor instance for encoding.
    """

    def __init__(self, dataframe, preprocessor):
        self.data = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor
        logger.info("Created dataset with %d samples", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq_encoded, mask = self.preprocessor.encode_sequence(row["seq"])
        sst8_encoded = self.preprocessor.encode_labels(row["sst8"], "sst8")
        sst3_encoded = self.preprocessor.encode_labels(row["sst3"], "sst3")

        return {
            "input_ids": torch.tensor(seq_encoded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "sst8_labels": torch.tensor(sst8_encoded, dtype=torch.long),
            "sst3_labels": torch.tensor(sst3_encoded, dtype=torch.long),
        }


def create_data_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split a DataFrame into train, validation, and test sets.

    Args:
        df: Input DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]

    logger.info("Data split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df
