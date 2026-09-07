"""Smoke tests for the PyTorch-based deep embedding models.

Each model is instantiated with a small config and run on a tiny synthetic
batch of shape [n_samples, seq_len, n_features] - the same layout
`PatternAnalysis` feeds them. These only check that construction, a
forward/encode pass, and one training epoch complete without error and
produce the expected shapes; they say nothing about representation quality.

Skipped entirely when torch is not installed (e.g. a tensorflow-only
environment), mirroring the module-load split used on the HPC cluster
(see hpc/smoke_benchmark_autoencoders.sh).
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datahandlers import ImputationDataset, PacmanDataset  # noqa: E402
from src.models import (  # noqa: E402
    AE_Trainer,
    AELSTM,
    TimeVAE,
    Transformer_Trainer,
    TSTransformerEncoder,
    VAE_Trainer,
    VanillaVAE,
)

N_SAMPLES = 10
SEQ_LEN = 12
N_FEATURES = 3
LATENT_DIM = 4


def make_synthetic_sequences(n=N_SAMPLES, seq_len=SEQ_LEN, n_features=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, seq_len, n_features)).astype(np.float32)


class TestAELSTM:
    def test_forward_and_encode_shapes(self):
        data = torch.from_numpy(make_synthetic_sequences())
        model = AELSTM(input_size=N_FEATURES, hidden_size=LATENT_DIM, dropout=0.0)

        reconstruction = model(data)
        assert reconstruction.shape == data.shape

        encoding = model.encode(data)
        assert encoding.shape == (N_SAMPLES, LATENT_DIM)

    def test_trains_one_epoch_without_error(self):
        dataset = PacmanDataset(gamestates=make_synthetic_sequences())
        model = AELSTM(input_size=N_FEATURES, hidden_size=LATENT_DIM, dropout=0.0)
        trainer = AE_Trainer(max_epochs=1, batch_size=4, validation_split=0.3, verbose=False)

        trainer.fit(model, dataset)

        assert len(trainer.train_loss_list) == 1
        assert math.isfinite(trainer.train_loss_list[-1])


class TestTSTransformerEncoder:
    def _build_model(self):
        return TSTransformerEncoder(
            feat_dim=N_FEATURES,
            max_len=SEQ_LEN,
            d_model=8,
            n_heads=2,
            num_layers=1,
            dim_feedforward=16,
        )

    def test_forward_and_encode_shapes(self):
        data = torch.from_numpy(make_synthetic_sequences())
        padding_mask = torch.ones(N_SAMPLES, SEQ_LEN, dtype=torch.bool)
        model = self._build_model()

        reconstruction = model(data, padding_mask)
        assert reconstruction.shape == data.shape

        pooled = model.encode(data, padding_mask, pooling=True)
        assert pooled.shape == (N_SAMPLES, 8)

    def test_trains_one_epoch_without_error(self):
        dataset = ImputationDataset(gamestates=make_synthetic_sequences())
        model = self._build_model()
        trainer = Transformer_Trainer(max_epochs=1, batch_size=4, validation_split=0.3, verbose=False)

        trainer.fit(model, dataset)

        assert len(trainer.train_loss_list) == 1
        assert math.isfinite(trainer.train_loss_list[-1])


class TestVanillaVAE:
    @pytest.mark.parametrize("pooling", [False, True])
    def test_forward_shapes(self, pooling):
        data = torch.from_numpy(make_synthetic_sequences())
        padding_mask = torch.ones(N_SAMPLES, SEQ_LEN) if pooling else None
        model = VanillaVAE(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[8, 16],
            pooling=pooling,
        )

        recon, mu, log_var = model(data, padding_mask=padding_mask)

        assert recon.shape == data.shape
        assert mu.shape == (N_SAMPLES, LATENT_DIM)
        assert log_var.shape == (N_SAMPLES, LATENT_DIM)

    def test_trains_one_epoch_without_error(self):
        dataset = ImputationDataset(gamestates=make_synthetic_sequences())
        model = VanillaVAE(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[8, 16],
            pooling=True,
        )
        trainer = VAE_Trainer(max_epochs=1, batch_size=4, validation_split=0.3, verbose=False)

        trainer.fit(model, dataset)

        assert len(trainer.train_loss_list) == 1
        assert math.isfinite(trainer.train_loss_list[-1])


class TestTimeVAE:
    @pytest.mark.parametrize("pooling", [False, True])
    def test_forward_shapes(self, pooling):
        data = torch.from_numpy(make_synthetic_sequences())
        padding_mask = torch.ones(N_SAMPLES, SEQ_LEN) if pooling else None
        model = TimeVAE(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_layer_sizes=[8, 16],
            pooling=pooling,
        )

        recon, mu, log_var = model(data, padding_mask=padding_mask)

        assert recon.shape == data.shape
        assert mu.shape == (N_SAMPLES, LATENT_DIM)
        assert log_var.shape == (N_SAMPLES, LATENT_DIM)

    def test_trains_one_epoch_without_error(self):
        dataset = ImputationDataset(gamestates=make_synthetic_sequences())
        model = TimeVAE(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_layer_sizes=[8, 16],
            pooling=True,
        )
        trainer = VAE_Trainer(max_epochs=1, batch_size=4, validation_split=0.3, verbose=False)

        trainer.fit(model, dataset)

        assert len(trainer.train_loss_list) == 1
        assert math.isfinite(trainer.train_loss_list[-1])
