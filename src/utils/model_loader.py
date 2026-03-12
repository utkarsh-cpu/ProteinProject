"""Model loader utility for loading pretrained weights and configurations.

Supports:
- Loading saved model weights from checkpoints
- Loading preprocessor (tokenizer/encoder) configuration
- Loading model configuration
- Skipping training and directly running inference
"""

import logging
import os

import torch

from src.model.config import ModelConfig
from src.model.transformer import ProteinTransformer
from src.data.preprocessing import ProteinPreprocessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility for loading pretrained protein structure prediction models.

    Example usage:
        loader = ModelLoader("outputs")
        model, preprocessor = loader.load()
        # model is ready for inference
    """

    def __init__(self, model_dir="outputs"):
        """Initialize the model loader.

        Args:
            model_dir: Directory containing saved model files
                       (checkpoint_best.pt, model_config.json, preprocessor.json).
        """
        self.model_dir = model_dir

    def load(self, checkpoint_name="checkpoint_best.pt", device=None):
        """Load a pretrained model, config, and preprocessor.

        Args:
            checkpoint_name: Name of the checkpoint file.
            device: Torch device. Auto-detected if None.

        Returns:
            Tuple of (model, preprocessor).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(self.model_dir, "model_config.json")
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.json")
        checkpoint_path = os.path.join(self.model_dir, checkpoint_name)

        logger.info("Loading model config from %s", config_path)
        config = ModelConfig.load(config_path)

        logger.info("Loading preprocessor from %s", preprocessor_path)
        preprocessor = ProteinPreprocessor.load(preprocessor_path)

        logger.info("Building model...")
        model = ProteinTransformer(config)

        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
        logger.info("Model loaded (epoch=%s, val_loss=%s)", epoch, val_loss)

        return model, preprocessor

    def load_checkpoint_info(self, checkpoint_name="checkpoint_best.pt"):
        """Load metadata from a checkpoint without building the model.

        Args:
            checkpoint_name: Name of the checkpoint file.

        Returns:
            Dictionary with checkpoint metadata.
        """
        path = os.path.join(self.model_dir, checkpoint_name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "history": checkpoint.get("history"),
        }
