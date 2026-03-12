"""Dataset download utility for protein secondary structure data.

Supports downloading from Kaggle or generating synthetic data for testing.
"""

import os
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
NONSTD_AA_CHAR = "*"
SST8_CLASSES = list("CEHBGITS")
SST3_CLASSES = list("CEH")

SST8_TO_SST3 = {
    "C": "C", "S": "C", "T": "C",
    "E": "E", "B": "E",
    "H": "H", "G": "H", "I": "H",
}


def download_dataset(data_dir="data/raw", kaggle_dataset="alfrandom/protein-secondary-structure"):
    """Download the protein secondary structure dataset from Kaggle.

    Args:
        data_dir: Directory to save downloaded files.
        kaggle_dataset: Kaggle dataset identifier.

    Returns:
        Path to the downloaded CSV file, or None if download fails.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "protein_secondary_structure.csv")

    if os.path.exists(csv_path):
        logger.info("Dataset already exists at %s", csv_path)
        return csv_path

    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(kaggle_dataset, path=data_dir, unzip=True)
        logger.info("Dataset downloaded to %s", data_dir)

        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                downloaded = os.path.join(data_dir, f)
                if downloaded != csv_path:
                    os.rename(downloaded, csv_path)
                return csv_path

    except (ImportError, Exception) as e:
        logger.warning("Kaggle download failed (%s). Generating synthetic dataset.", e)
        return None

    return csv_path


def generate_synthetic_dataset(num_samples=500, min_len=30, max_len=150, seed=42):
    """Generate a synthetic protein dataset for testing.

    Args:
        num_samples: Number of protein sequences to generate.
        min_len: Minimum sequence length.
        max_len: Maximum sequence length.
        seed: Random seed for reproducibility.

    Returns:
        pandas DataFrame with columns matching the real dataset.
    """
    rng = np.random.RandomState(seed)
    records = []

    for i in range(num_samples):
        seq_len = rng.randint(min_len, max_len + 1)
        has_nonstd = rng.random() < 0.1

        seq = list(rng.choice(STANDARD_AMINO_ACIDS, size=seq_len))
        if has_nonstd:
            n_nonstd = rng.randint(1, max(2, seq_len // 20))
            positions = rng.choice(seq_len, size=n_nonstd, replace=False)
            for pos in positions:
                seq[pos] = NONSTD_AA_CHAR

        sst8 = list(rng.choice(SST8_CLASSES, size=seq_len, p=[0.30, 0.20, 0.25, 0.03, 0.05, 0.01, 0.08, 0.08]))
        sst3 = [SST8_TO_SST3[s] for s in sst8]

        records.append({
            "pdb_id": f"SYN{i:04d}",
            "chain_code": "A",
            "seq": "".join(seq),
            "sst8": "".join(sst8),
            "sst3": "".join(sst3),
            "len": seq_len,
            "has_nonstd_aa": has_nonstd,
        })

    return pd.DataFrame(records)
