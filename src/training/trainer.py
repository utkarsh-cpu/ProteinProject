"""Training pipeline for the protein secondary structure prediction model.

Provides a modular Trainer class that handles:
- Data loading with PyTorch DataLoaders
- Model initialization
- Training loop with gradient accumulation
- Validation loop with metric computation
- Checkpointing (saving best and latest models)
"""

import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for the ProteinTransformer model.

    Args:
        model: ProteinTransformer model instance.
        train_dataset: Training ProteinDataset.
        val_dataset: Validation ProteinDataset.
        config: Dictionary of training hyperparameters.
        device: Torch device to use.
        output_dir: Directory for saving checkpoints and logs.
    """

    def __init__(self, model, train_dataset, val_dataset, config=None, device=None,
                 output_dir="outputs"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        defaults = {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": 10,
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
            "patience": 5,
        }
        self.config = {**defaults, **(config or {})}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc_sst8": [], "val_acc_sst8": [],
            "train_acc_sst3": [], "val_acc_sst3": [],
        }

    def train(self):
        """Run the full training loop.

        Returns:
            Dictionary with training history.
        """
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config["batch_size"],
            shuffle=True, num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config["batch_size"],
            shuffle=False, num_workers=0, pin_memory=False,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training for %d epochs on %s", self.config["num_epochs"], self.device)

        for epoch in range(1, self.config["num_epochs"] + 1):
            start_time = time.time()

            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc_sst8"].append(train_metrics["acc_sst8"])
            self.history["val_acc_sst8"].append(val_metrics["acc_sst8"])
            self.history["train_acc_sst3"].append(train_metrics["acc_sst3"])
            self.history["val_acc_sst3"].append(val_metrics["acc_sst3"])

            self.scheduler.step(val_metrics["loss"])

            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d [%.1fs] - Train Loss: %.4f, Val Loss: %.4f, "
                "Val Acc Q8: %.4f, Val Acc Q3: %.4f",
                epoch, self.config["num_epochs"], elapsed,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["acc_sst8"], val_metrics["acc_sst3"],
            )

            self._save_checkpoint(epoch, val_metrics["loss"], "latest")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics["loss"], "best")
                logger.info("New best model saved (val_loss=%.4f)", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info("Early stopping triggered after %d epochs", epoch)
                    break

        return self.history

    def _train_epoch(self, loader, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct_sst8 = 0
        total_correct_sst3 = 0
        total_tokens = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sst8_labels=batch["sst8_labels"],
                sst3_labels=batch["sst3_labels"],
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()

            mask = batch["attention_mask"].bool()
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            sst8_preds = self._get_predictions(outputs["sst8_preds"], mask)
            sst3_preds = self._get_predictions(outputs["sst3_preds"], mask)
            total_correct_sst8 += (sst8_preds == batch["sst8_labels"][mask]).sum().item()
            total_correct_sst3 += (sst3_preds == batch["sst3_labels"][mask]).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / max(total_tokens, 1),
            "acc_sst8": total_correct_sst8 / max(total_tokens, 1),
            "acc_sst3": total_correct_sst3 / max(total_tokens, 1),
        }

    def _validate_epoch(self, loader, epoch):
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct_sst8 = 0
        total_correct_sst3 = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Val Epoch {epoch}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sst8_labels=batch["sst8_labels"],
                    sst3_labels=batch["sst3_labels"],
                )

                mask = batch["attention_mask"].bool()
                n_tokens = mask.sum().item()
                total_loss += outputs["loss"].item() * n_tokens
                total_tokens += n_tokens

                sst8_preds = self._get_predictions(outputs["sst8_preds"], mask)
                sst3_preds = self._get_predictions(outputs["sst3_preds"], mask)
                total_correct_sst8 += (sst8_preds == batch["sst8_labels"][mask]).sum().item()
                total_correct_sst3 += (sst3_preds == batch["sst3_labels"][mask]).sum().item()

        return {
            "loss": total_loss / max(total_tokens, 1),
            "acc_sst8": total_correct_sst8 / max(total_tokens, 1),
            "acc_sst3": total_correct_sst3 / max(total_tokens, 1),
        }

    def _get_predictions(self, preds, mask):
        """Extract predictions for non-padded tokens.

        Handles both CRF (list of lists) and softmax (tensor) predictions.
        """
        if isinstance(preds, list):
            flat = []
            for i, seq_preds in enumerate(preds):
                seq_mask = mask[i]
                n_real = seq_mask.sum().item()
                flat.extend(seq_preds[:n_real])
            return torch.tensor(flat, device=mask.device)
        else:
            return preds[mask]

    def _save_checkpoint(self, epoch, val_loss, tag="latest"):
        """Save a model checkpoint."""
        path = os.path.join(self.output_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }, path)
