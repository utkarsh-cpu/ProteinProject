"""Tests for the protein secondary structure prediction pipeline.

Tests cover:
- Data preprocessing (encoding/decoding)
- Synthetic data generation
- Model forward pass (with and without CRF)
- CRF layer (loss computation and decoding)
- Training loop (single epoch)
- Evaluation metrics
- Model save/load cycle
- Inference utility
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.download import generate_synthetic_dataset
from src.data.preprocessing import ProteinPreprocessor
from src.data.dataset import ProteinDataset, create_data_splits
from src.model.config import ModelConfig
from src.model.transformer import ProteinTransformer
from src.model.crf import CRFLayer
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator
from src.utils.model_loader import ModelLoader
from src.utils.inference import predict_structure


@pytest.fixture
def preprocessor():
    return ProteinPreprocessor(max_seq_len=64)


@pytest.fixture
def synthetic_df():
    return generate_synthetic_dataset(num_samples=50, min_len=20, max_len=60, seed=42)


@pytest.fixture
def model_config(preprocessor):
    return ModelConfig(
        vocab_size=preprocessor.vocab_size,
        num_sst8_classes=preprocessor.num_sst8_classes,
        num_sst3_classes=preprocessor.num_sst3_classes,
        max_seq_len=64,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        use_crf=True,
    )


@pytest.fixture
def model(model_config):
    return ProteinTransformer(model_config)


class TestPreprocessor:
    """Tests for ProteinPreprocessor."""

    def test_encode_sequence(self, preprocessor):
        seq = "ACDEFG"
        encoded, mask = preprocessor.encode_sequence(seq)
        assert len(encoded) == 64
        assert len(mask) == 64
        assert mask[:6].sum() == 6
        assert mask[6:].sum() == 0
        assert encoded[0] == preprocessor.aa_to_idx["A"]

    def test_encode_labels(self, preprocessor):
        labels = "CCHHEE"
        encoded = preprocessor.encode_labels(labels, "sst8")
        assert len(encoded) == 64
        assert encoded[0] == preprocessor.sst8_to_idx["C"]
        assert encoded[6:].sum() == 0

    def test_decode_sequence(self, preprocessor):
        seq = "ACDEFG"
        encoded, mask = preprocessor.encode_sequence(seq)
        decoded = preprocessor.decode_sequence(encoded, mask)
        assert decoded == seq

    def test_decode_labels(self, preprocessor):
        labels = "CCHHEE"
        encoded = preprocessor.encode_labels(labels, "sst8")
        mask = np.array([1] * 6 + [0] * 58, dtype=np.int64)
        decoded = preprocessor.decode_labels(encoded, mask, "sst8")
        assert decoded == labels

    def test_nonstd_amino_acid(self, preprocessor):
        seq = "AC*EF"
        encoded, mask = preprocessor.encode_sequence(seq)
        assert encoded[2] == preprocessor.aa_to_idx["*"]

    def test_truncation(self, preprocessor):
        seq = "A" * 100
        encoded, mask = preprocessor.encode_sequence(seq)
        assert len(encoded) == 64
        assert mask.sum() == 64

    def test_save_load(self, preprocessor):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "preprocessor.json")
            preprocessor.save(path)
            loaded = ProteinPreprocessor.load(path)
            assert loaded.vocab_size == preprocessor.vocab_size
            assert loaded.max_seq_len == preprocessor.max_seq_len
            assert loaded.aa_to_idx == preprocessor.aa_to_idx


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_generate(self, synthetic_df):
        assert len(synthetic_df) == 50
        assert set(synthetic_df.columns) >= {"seq", "sst8", "sst3", "len", "pdb_id"}

    def test_sequence_lengths(self, synthetic_df):
        for _, row in synthetic_df.iterrows():
            assert len(row["seq"]) == row["len"]
            assert len(row["sst8"]) == row["len"]
            assert len(row["sst3"]) == row["len"]

    def test_data_split(self, synthetic_df):
        train, val, test = create_data_splits(synthetic_df)
        assert len(train) + len(val) + len(test) == len(synthetic_df)
        assert len(train) > len(val)
        assert len(train) > len(test)


class TestDataset:
    """Tests for ProteinDataset."""

    def test_getitem(self, synthetic_df, preprocessor):
        dataset = ProteinDataset(synthetic_df, preprocessor)
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "sst8_labels" in sample
        assert "sst3_labels" in sample
        assert sample["input_ids"].shape == (64,)

    def test_len(self, synthetic_df, preprocessor):
        dataset = ProteinDataset(synthetic_df, preprocessor)
        assert len(dataset) == 50


class TestCRFLayer:
    """Tests for the CRF layer."""

    def test_forward(self):
        crf = CRFLayer(num_tags=4, pad_idx=0)
        emissions = torch.randn(2, 10, 4)
        tags = torch.randint(1, 4, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[1, 7:] = 0
        loss = crf(emissions, tags, mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_decode(self):
        crf = CRFLayer(num_tags=4, pad_idx=0)
        emissions = torch.randn(2, 10, 4)
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[1, 7:] = 0
        best_tags = crf.decode(emissions, mask)
        assert len(best_tags) == 2
        assert len(best_tags[0]) == 10
        assert len(best_tags[1]) == 7


class TestTransformerModel:
    """Tests for the ProteinTransformer model."""

    def test_forward_with_labels(self, model, synthetic_df, preprocessor):
        dataset = ProteinDataset(synthetic_df, preprocessor)
        sample = dataset[0]
        batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        outputs = model(**batch)
        assert "loss" in outputs
        assert "sst8_logits" in outputs
        assert "sst3_logits" in outputs
        assert "sst8_preds" in outputs
        assert "sst3_preds" in outputs

    def test_forward_without_labels(self, model, synthetic_df, preprocessor):
        dataset = ProteinDataset(synthetic_df, preprocessor)
        sample = dataset[0]
        outputs = model(
            input_ids=sample["input_ids"].unsqueeze(0),
            attention_mask=sample["attention_mask"].unsqueeze(0),
        )
        assert "loss" not in outputs
        assert "sst8_preds" in outputs

    def test_no_crf_model(self, preprocessor):
        config = ModelConfig(
            vocab_size=preprocessor.vocab_size,
            num_sst8_classes=preprocessor.num_sst8_classes,
            num_sst3_classes=preprocessor.num_sst3_classes,
            max_seq_len=64, d_model=32, nhead=4, num_encoder_layers=2,
            dim_feedforward=64, use_crf=False,
        )
        model = ProteinTransformer(config)
        input_ids = torch.randint(0, preprocessor.vocab_size, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=mask)
        assert outputs["sst8_preds"].shape == (2, 64)


class TestModelConfig:
    """Tests for ModelConfig save/load."""

    def test_save_load(self):
        config = ModelConfig(d_model=64, nhead=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save(path)
            loaded = ModelConfig.load(path)
            assert loaded.d_model == 64
            assert loaded.nhead == 4


class TestTrainer:
    """Tests for the Trainer (single epoch)."""

    def test_train_one_epoch(self, model, synthetic_df, preprocessor):
        train_df, val_df, _ = create_data_splits(synthetic_df)
        train_ds = ProteinDataset(train_df, preprocessor)
        val_ds = ProteinDataset(val_df, preprocessor)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config={"batch_size": 8, "num_epochs": 1, "learning_rate": 1e-3},
                output_dir=tmpdir,
            )
            history = trainer.train()
            assert len(history["train_loss"]) == 1
            assert len(history["val_loss"]) == 1
            assert os.path.exists(os.path.join(tmpdir, "checkpoint_latest.pt"))


class TestEvaluator:
    """Tests for the Evaluator."""

    def test_evaluate(self, model, synthetic_df, preprocessor):
        _, _, test_df = create_data_splits(synthetic_df)
        test_ds = ProteinDataset(test_df, preprocessor)
        evaluator = Evaluator(model, preprocessor)
        metrics = evaluator.evaluate(test_ds, batch_size=8)
        assert "sst8" in metrics
        assert "sst3" in metrics
        assert "accuracy" in metrics["sst8"]
        assert "f1" in metrics["sst8"]
        assert "confusion_matrix" in metrics["sst8"]

    def test_save_report(self, model, synthetic_df, preprocessor):
        _, _, test_df = create_data_splits(synthetic_df)
        test_ds = ProteinDataset(test_df, preprocessor)
        evaluator = Evaluator(model, preprocessor)
        metrics = evaluator.evaluate(test_ds, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            evaluator.save_report(metrics, path)
            assert os.path.exists(path)


class TestModelLoader:
    """Tests for model save/load cycle."""

    def test_save_and_load(self, model, model_config, preprocessor):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_config.save(os.path.join(tmpdir, "model_config.json"))
            preprocessor.save(os.path.join(tmpdir, "preprocessor.json"))
            torch.save(
                {"epoch": 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": {}, "val_loss": 1.0, "history": {}},
                os.path.join(tmpdir, "checkpoint_best.pt"),
            )

            loader = ModelLoader(tmpdir)
            loaded_model, loaded_preprocessor = loader.load()
            assert loaded_preprocessor.vocab_size == preprocessor.vocab_size

            model.eval()
            input_ids = torch.randint(0, preprocessor.vocab_size, (1, 64))
            mask = torch.ones(1, 64, dtype=torch.long)
            with torch.no_grad():
                out1 = model(input_ids=input_ids, attention_mask=mask)
                out2 = loaded_model(input_ids=input_ids, attention_mask=mask)
            assert torch.allclose(out1["sst8_logits"], out2["sst8_logits"])


class TestInference:
    """Tests for the inference utility."""

    def test_predict_structure(self, model, preprocessor):
        result = predict_structure(
            "ACDEFGHIKLMNPQRSTVWY",
            model=model, preprocessor=preprocessor,
        )
        assert result["length"] == 20
        assert len(result["sst8"]) == 20
        assert len(result["sst3"]) == 20
        assert result["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
