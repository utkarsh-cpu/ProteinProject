"""Main training script for the protein secondary structure prediction pipeline.

Usage:
    python train.py [--data_path PATH] [--output_dir DIR] [--epochs N] [--batch_size N]
                    [--d_model N] [--nhead N] [--num_layers N] [--max_seq_len N]
                    [--lr FLOAT] [--no_crf] [--synthetic]

Examples:
    # Train with synthetic data (for testing):
    python train.py --synthetic --epochs 3

    # Train with real Kaggle data:
    python train.py --data_path data/raw/protein_secondary_structure.csv --epochs 20

    # Train with custom hyperparameters:
    python train.py --synthetic --d_model 256 --nhead 8 --num_layers 6 --epochs 10
"""

import argparse
import logging
import os
import sys

import pandas as pd
import torch

from src.data.download import download_dataset, generate_synthetic_dataset
from src.data.preprocessing import ProteinPreprocessor
from src.data.dataset import ProteinDataset, create_data_splits
from src.model.config import ModelConfig
from src.model.transformer import ProteinTransformer
from src.training.trainer import Trainer
from src.visualization.plots import plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train protein structure prediction model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for model and logs")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no_crf", action="store_true", help="Disable CRF layer")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data Loading ----
    if args.synthetic:
        logger.info("Generating synthetic dataset with %d samples", args.num_samples)
        df = generate_synthetic_dataset(num_samples=args.num_samples)
    elif args.data_path and os.path.exists(args.data_path):
        logger.info("Loading dataset from %s", args.data_path)
        df = pd.read_csv(args.data_path)
    else:
        logger.info("Attempting to download dataset from Kaggle...")
        csv_path = download_dataset()
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            logger.info("Download failed. Falling back to synthetic data.")
            df = generate_synthetic_dataset(num_samples=args.num_samples)

    required_cols = {"seq", "sst8", "sst3"}
    if not required_cols.issubset(set(df.columns)):
        logger.error("Dataset must contain columns: %s. Found: %s", required_cols, list(df.columns))
        sys.exit(1)

    df = df.dropna(subset=["seq", "sst8", "sst3"]).reset_index(drop=True)
    logger.info("Dataset loaded: %d sequences", len(df))

    # ---- Preprocessing ----
    preprocessor = ProteinPreprocessor(max_seq_len=args.max_seq_len)

    # ---- Data Split ----
    train_df, val_df, test_df = create_data_splits(df)

    train_dataset = ProteinDataset(train_df, preprocessor)
    val_dataset = ProteinDataset(val_df, preprocessor)
    test_dataset = ProteinDataset(test_df, preprocessor)

    # ---- Model Config ----
    model_config = ModelConfig(
        vocab_size=preprocessor.vocab_size,
        num_sst8_classes=preprocessor.num_sst8_classes,
        num_sst3_classes=preprocessor.num_sst3_classes,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_crf=not args.no_crf,
    )

    # ---- Save configs ----
    model_config.save(os.path.join(args.output_dir, "model_config.json"))
    preprocessor.save(os.path.join(args.output_dir, "preprocessor.json"))
    logger.info("Configs saved to %s", args.output_dir)

    # ---- Model ----
    model = ProteinTransformer(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f MB)", total_params, total_params * 4 / 1e6)

    # ---- Training ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_epochs": args.epochs,
            "patience": args.patience,
        },
        device=device,
        output_dir=args.output_dir,
    )

    history = trainer.train()

    # ---- Plots ----
    plot_dir = os.path.join(args.output_dir, "plots")
    plot_training_curves(history, output_dir=plot_dir)

    logger.info("Training complete. Outputs saved to %s", args.output_dir)
    logger.info("To evaluate the trained model, run: python evaluate.py --output_dir %s", args.output_dir)


if __name__ == "__main__":
    main()
