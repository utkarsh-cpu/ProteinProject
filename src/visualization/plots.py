"""Visualization utilities for training diagnostics and model evaluation.

Generates plots for:
- Training vs validation loss curves
- Accuracy curves for Q8 and Q3
- Confusion matrices
- Prediction examples
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_training_curves(history, output_dir="outputs/plots"):
    """Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss',
                 'train_acc_sst8', 'val_acc_sst8', 'train_acc_sst3', 'val_acc_sst3'.
        output_dir: Directory to save plot files.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc_sst8"], "b-o", label="Train Q8 Acc")
    axes[1].plot(epochs, history["val_acc_sst8"], "r-o", label="Val Q8 Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Q8 (8-State) Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["train_acc_sst3"], "b-o", label="Train Q3 Acc")
    axes[2].plot(epochs, history["val_acc_sst3"], "r-o", label="Val Q3 Acc")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Q3 (3-State) Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", path)


def plot_confusion_matrices(metrics, output_dir="outputs/plots"):
    """Plot confusion matrices for Q8 and Q3 predictions.

    Args:
        metrics: Dictionary from Evaluator.evaluate() with 'sst8' and 'sst3' keys.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_type in ["sst8", "sst3"]:
        cm = np.array(metrics[label_type]["confusion_matrix"])
        names = metrics[label_type]["label_names"]

        fig, ax = plt.subplots(figsize=(8, 6) if label_type == "sst8" else (6, 5))

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm / row_sums

        sns.heatmap(
            cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=names, yticklabels=names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = "Q8 Confusion Matrix" if label_type == "sst8" else "Q3 Confusion Matrix"
        ax.set_title(title)
        plt.tight_layout()

        path = os.path.join(output_dir, f"confusion_matrix_{label_type}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix (%s) saved to %s", label_type, path)


def plot_prediction_examples(model, dataset, preprocessor, num_examples=5,
                              output_dir="outputs/plots", device=None):
    """Plot prediction examples comparing true vs predicted structures.

    Args:
        model: Trained ProteinTransformer model.
        dataset: ProteinDataset instance.
        preprocessor: ProteinPreprocessor for decoding.
        num_examples: Number of examples to plot.
        output_dir: Directory to save plots.
        device: Torch device.
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()

    num_examples = min(num_examples, len(dataset))

    fig, axes = plt.subplots(num_examples, 1, figsize=(14, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    for i in range(num_examples):
        sample = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        mask = attention_mask[0].bool().cpu()
        n_real = mask.sum().item()

        true_sst3 = sample["sst3_labels"][:n_real].numpy()

        if isinstance(outputs["sst3_preds"], list):
            pred_sst3 = np.array(outputs["sst3_preds"][0][:n_real])
        else:
            pred_sst3 = outputs["sst3_preds"][0, :n_real].cpu().numpy()

        display_len = min(n_real, 80)

        ax = axes[i]
        x = np.arange(display_len)
        width = 0.35

        ax.bar(x - width / 2, true_sst3[:display_len], width, label="True", alpha=0.7)
        ax.bar(x + width / 2, pred_sst3[:display_len], width, label="Predicted", alpha=0.7)
        ax.set_ylabel("Structure Class")
        ax.set_title(f"Example {i + 1} - Q3 Prediction (first {display_len} residues)")

        sst3_names = ["C", "E", "H"]
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(sst3_names)
        ax.legend(loc="upper right")

        seq = preprocessor.decode_sequence(sample["input_ids"][:display_len].numpy())
        if display_len <= 40:
            ax.set_xticks(x)
            ax.set_xticklabels(list(seq), fontsize=6)

    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction examples saved to %s", path)
