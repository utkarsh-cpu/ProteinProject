"""Evaluation metrics and report generation for protein structure prediction.

Computes accuracy, precision, recall, F1 score, and confusion matrices for
both 8-state (Q8) and 3-state (Q3) secondary structure predictions.
"""

import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a trained model on a test dataset.

    Args:
        model: Trained ProteinTransformer model.
        preprocessor: ProteinPreprocessor for decoding labels.
        device: Torch device.
    """

    def __init__(self, model, preprocessor, device=None):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataset, batch_size=32):
        """Evaluate the model on a dataset.

        Args:
            dataset: ProteinDataset to evaluate on.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with evaluation metrics for both Q8 and Q3.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_sst8_preds = []
        all_sst8_labels = []
        all_sst3_preds = []
        all_sst3_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                mask = batch["attention_mask"].bool()

                sst8_preds = self._extract_preds(outputs["sst8_preds"], mask)
                sst3_preds = self._extract_preds(outputs["sst3_preds"], mask)

                for i in range(mask.shape[0]):
                    seq_mask = mask[i]
                    n_real = seq_mask.sum().item()

                    sst8_labels_i = batch["sst8_labels"][i][:n_real].cpu().numpy()
                    sst3_labels_i = batch["sst3_labels"][i][:n_real].cpu().numpy()

                    all_sst8_labels.extend(sst8_labels_i.tolist())
                    all_sst3_labels.extend(sst3_labels_i.tolist())

                all_sst8_preds.extend(sst8_preds)
                all_sst3_preds.extend(sst3_preds)

        sst8_metrics = self._compute_metrics(
            all_sst8_labels, all_sst8_preds, "sst8",
            list(range(1, self.preprocessor.num_sst8_classes)),
            [self.preprocessor.idx_to_sst8.get(i, "?")
             for i in range(1, self.preprocessor.num_sst8_classes)],
        )
        sst3_metrics = self._compute_metrics(
            all_sst3_labels, all_sst3_preds, "sst3",
            list(range(1, self.preprocessor.num_sst3_classes)),
            [self.preprocessor.idx_to_sst3.get(i, "?")
             for i in range(1, self.preprocessor.num_sst3_classes)],
        )

        return {"sst8": sst8_metrics, "sst3": sst3_metrics}

    def _extract_preds(self, preds, mask):
        """Extract flat predictions for non-pad positions."""
        if isinstance(preds, list):
            flat = []
            for i, seq_preds in enumerate(preds):
                n_real = mask[i].sum().item()
                flat.extend(seq_preds[:n_real])
            return flat
        else:
            return preds[mask].cpu().numpy().tolist()

    def _compute_metrics(self, labels, preds, name, label_indices, label_names):
        """Compute classification metrics."""
        labels = np.array(labels)
        preds = np.array(preds)

        valid = np.isin(labels, label_indices)
        labels = labels[valid]
        preds = preds[valid]

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, labels=label_indices,
                                     average="macro", zero_division=0)
        recall = recall_score(labels, preds, labels=label_indices,
                               average="macro", zero_division=0)
        f1 = f1_score(labels, preds, labels=label_indices,
                       average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=label_indices)

        report = classification_report(
            labels, preds, labels=label_indices,
            target_names=label_names, zero_division=0,
        )

        logger.info("%s - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                     name.upper(), acc, precision, recall, f1)
        logger.info("\n%s Classification Report:\n%s", name.upper(), report)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "label_names": label_names,
        }

    def save_report(self, metrics, path):
        """Save evaluation report to a JSON file.

        Args:
            metrics: Dictionary of evaluation metrics.
            path: File path for the report.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        report = {
            "sst8": {k: v for k, v in metrics["sst8"].items()
                     if k != "classification_report"},
            "sst3": {k: v for k, v in metrics["sst3"].items()
                     if k != "classification_report"},
            "sst8_report": metrics["sst8"]["classification_report"],
            "sst3_report": metrics["sst3"]["classification_report"],
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", path)
