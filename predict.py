"""Prediction/inference script for protein secondary structure.

Usage:
    python predict.py --sequence SEQUENCE [--output_dir DIR]

Examples:
    python predict.py --sequence "MKFLILLFNILCLFPVLAADNHGVS"
    python predict.py --sequence "ACDEFGHIKLMNPQRSTVWY" --output_dir outputs
"""

import argparse
import json
import logging

from src.utils.inference import predict_structure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict protein secondary structure")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Amino acid sequence string")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory with saved model")
    return parser.parse_args()


def main():
    args = parse_args()

    result = predict_structure(args.sequence, model_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("Protein Secondary Structure Prediction")
    print("=" * 60)
    print(f"Sequence ({result['length']} residues):")
    print(f"  {result['sequence']}")
    print(f"\nQ8 (8-state) prediction:")
    print(f"  {result['sst8']}")
    print(f"\nQ3 (3-state) prediction:")
    print(f"  {result['sst3']}")
    print("=" * 60)

    logger.info("Prediction result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
