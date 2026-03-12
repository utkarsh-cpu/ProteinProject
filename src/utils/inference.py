"""Inference utility for predicting secondary structure of new protein sequences.

Provides a simple interface for running predictions on new amino acid sequences
without requiring knowledge of the underlying model architecture.
"""

import logging

import torch

from src.utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


def predict_structure(sequence, model=None, preprocessor=None, model_dir="outputs",
                       device=None):
    """Predict secondary structure for a protein sequence.

    Args:
        sequence: Amino acid sequence string (e.g., 'MKFLILLFNILCLFPVLAADNH...').
        model: Optional pre-loaded ProteinTransformer model.
        preprocessor: Optional pre-loaded ProteinPreprocessor.
        model_dir: Directory with saved model files (used if model/preprocessor not provided).
        device: Torch device.

    Returns:
        Dictionary with:
            'sequence': Input amino acid sequence.
            'sst8': Predicted 8-state secondary structure string.
            'sst3': Predicted 3-state secondary structure string.
            'length': Sequence length.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or preprocessor is None:
        loader = ModelLoader(model_dir)
        model, preprocessor = loader.load(device=device)

    model.eval()

    seq_encoded, mask = preprocessor.encode_sequence(sequence)
    input_ids = torch.tensor(seq_encoded, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    mask_np = mask

    if isinstance(outputs["sst8_preds"], list):
        sst8_indices = outputs["sst8_preds"][0]
    else:
        sst8_indices = outputs["sst8_preds"][0].cpu().numpy().tolist()

    if isinstance(outputs["sst3_preds"], list):
        sst3_indices = outputs["sst3_preds"][0]
    else:
        sst3_indices = outputs["sst3_preds"][0].cpu().numpy().tolist()

    sst8_str = preprocessor.decode_labels(sst8_indices, mask_np, "sst8")
    sst3_str = preprocessor.decode_labels(sst3_indices, mask_np, "sst3")

    return {
        "sequence": sequence,
        "sst8": sst8_str,
        "sst3": sst3_str,
        "length": len(sequence),
    }
