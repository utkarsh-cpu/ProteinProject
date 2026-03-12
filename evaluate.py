"""Evaluation script for the protein secondary structure prediction model.

Usage:
    python evaluate.py [--data_path PATH] [--output_dir DIR] [--batch_size N] [--synthetic]

Examples:
    # Evaluate with synthetic data:
    python evaluate.py --synthetic --output_dir outputs

    # Evaluate with real data:
    python evaluate.py --data_path data/raw/protein_secondary_structure.csv --output_dir outputs
"""

import argparse
import logging
import os
import sys

import pandas as pd

from src.data.download import generate_synthetic_dataset
from src.data.preprocessing import ProteinPreprocessor
from src.data.dataset import ProteinDataset, create_data_splits
from src.evaluation.metrics import Evaluator
from src.utils.model_loader import ModelLoader
from src.visualization.plots import plot_confusion_matrices, plot_prediction_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate protein structure prediction model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory with saved model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load Model ----
    loader = ModelLoader(args.output_dir)
    model, preprocessor = loader.load()

    # ---- Load Data ----
    if args.synthetic:
        df = generate_synthetic_dataset(num_samples=args.num_samples)
    elif args.data_path and os.path.exists(args.data_path):
        df = pd.read_csv(args.data_path)
    else:
        logger.error("Provide --data_path or --synthetic")
        sys.exit(1)

    df = df.dropna(subset=["seq", "sst8", "sst3"]).reset_index(drop=True)
    _, _, test_df = create_data_splits(df)
    test_dataset = ProteinDataset(test_df, preprocessor)
    logger.info("Test set: %d samples", len(test_dataset))

    # ---- Evaluate ----
    evaluator = Evaluator(model, preprocessor)
    metrics = evaluator.evaluate(test_dataset, batch_size=args.batch_size)

    # ---- Save Report ----
    report_path = os.path.join(args.output_dir, "validation_report.json")
    evaluator.save_report(metrics, report_path)

    # ---- Plots ----
    plot_dir = os.path.join(args.output_dir, "plots")
    plot_confusion_matrices(metrics, output_dir=plot_dir)
    plot_prediction_examples(model, test_dataset, preprocessor, output_dir=plot_dir)

    logger.info("Evaluation complete. Report saved to %s", report_path)


if __name__ == "__main__":
    main()
