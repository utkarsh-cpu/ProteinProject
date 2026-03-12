# Protein Secondary Structure Prediction Pipeline

A Transformer-based deep learning pipeline for predicting protein secondary
structure (both 8-state Q8 and 3-state Q3) from amino acid sequences.

## Overview

This project implements an end-to-end machine learning pipeline that takes
raw protein amino acid sequences and predicts their secondary structure
assignments. The pipeline includes data preprocessing, model training,
evaluation, visualization, and inference utilities.

## Innovations from Context

The following novel approaches from `context.txt` are incorporated:

1. **Architectural Novelty — Transformer Self-Attention**: The core model uses
   a Transformer encoder with multi-head self-attention to capture long-range
   dependencies between distant amino acids, enabling the network to understand
   global folding structure across the entire sequence.

2. **Output Novelty — Conditional Random Field (CRF)**: An optional CRF layer
   replaces standard Softmax for sequence labeling. The CRF models transition
   probabilities between consecutive secondary structure states, preventing
   physically impossible predictions (e.g., a single-residue helix).

3. **Input Novelty — PLM Embedding Support**: The model architecture accepts
   external embeddings via the `external_embeddings` parameter, enabling
   integration with Protein Language Models (ESM-2, ProtBERT) that encode
   evolutionary and chemical context from millions of protein sequences.

## Project Structure

```
├── context.txt              # Research context and novelty ideas
├── requirements.txt         # Python dependencies
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── predict.py               # Inference entry point
├── src/
│   ├── data/
│   │   ├── download.py      # Dataset download and synthetic data generation
│   │   ├── preprocessing.py # Amino acid and label encoding/decoding
│   │   └── dataset.py       # PyTorch Dataset and data splitting
│   ├── model/
│   │   ├── config.py        # Model configuration dataclass
│   │   ├── crf.py           # CRF layer (Viterbi decoding, forward algorithm)
│   │   └── transformer.py   # Transformer encoder model
│   ├── training/
│   │   └── trainer.py       # Training loop with checkpointing
│   ├── evaluation/
│   │   └── metrics.py       # Accuracy, precision, recall, F1, confusion matrix
│   ├── visualization/
│   │   └── plots.py         # Loss curves, accuracy curves, confusion matrices
│   └── utils/
│       ├── model_loader.py  # Load pretrained models
│       └── inference.py     # Predict structure from a sequence string
└── tests/
    └── test_pipeline.py     # Unit tests for all components
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with synthetic data (for testing):
python train.py --synthetic --epochs 3

# Train with Kaggle dataset:
python train.py --data_path data/raw/protein_secondary_structure.csv --epochs 20

# Custom hyperparameters:
python train.py --synthetic --d_model 256 --nhead 8 --num_layers 6 --epochs 10
```

### Evaluation

```bash
python evaluate.py --synthetic --output_dir outputs
```

Generates a `validation_report.json` with accuracy, precision, recall, F1 scores,
and confusion matrices, plus plots saved to `outputs/plots/`.

### Inference

```bash
python predict.py --sequence "MKFLILLFNILCLFPVLAADNHGVS" --output_dir outputs
```

### Loading a Pretrained Model (Python API)

```python
from src.utils.model_loader import ModelLoader
from src.utils.inference import predict_structure

# Load model
loader = ModelLoader("outputs")
model, preprocessor = loader.load()

# Predict
result = predict_structure("ACDEFGHIKLMNPQRSTVWY", model=model, preprocessor=preprocessor)
print(result["sst3"])  # e.g., "CCHHHHHEEEEEEECCCCCC"
```

## Dataset

The pipeline supports the [Protein Secondary Structure dataset from Kaggle](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure).
The dataset contains protein sequences with columns: `pdb_id`, `chain_code`,
`seq`, `sst8`, `sst3`, `len`, and `has_nonstd_aa`. A synthetic data generator
is included for testing when the real dataset is unavailable.

## Model Architecture

- **Embedding Layer**: Learnable amino acid embeddings (or external PLM embeddings)
- **Positional Encoding**: Sinusoidal encoding for sequence position
- **Transformer Encoder**: Multi-head self-attention + feedforward layers
- **Dual Prediction Heads**: Separate linear heads for Q8 and Q3 prediction
- **CRF Layer** (optional): Captures label transition constraints

## Testing

```bash
python -m pytest tests/ -v
```
