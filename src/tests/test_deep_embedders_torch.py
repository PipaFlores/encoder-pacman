"""Smoke tests for the PyTorch-based deep embedding models.

Each model is instantiated with a small config and run on a tiny synthetic
batch of shape [n_samples, seq_len, n_features] - the same layout
`PatternAnalysis` feeds them. These only check that construction, a
forward/encode pass, and one training epoch complete without error and
produce the expected shapes; they say nothing about representation quality.

Skipped entirely when torch is not installed.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datahandlers import ImputationDataset, PacmanDataset
from src.models import (
    AELSTM,
    AE_Trainer,
    MLPAutoencoder,
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


class TestMLPAutoencoder:
    @pytest.mark.parametrize("pooling", [False, True])
    def test_forward_and_encode_shapes(self, pooling):
        data = torch.from_numpy(make_synthetic_sequences())
        lengths = torch.full((N_SAMPLES,), SEQ_LEN, dtype=torch.long)
        model = MLPAutoencoder(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[16, 8],
            pooling=pooling,
        )

        reconstruction = model(data, lengths=lengths)
        assert reconstruction.shape == data.shape

        encoding = model.encode(data, lengths=lengths)
        assert encoding.shape == (N_SAMPLES, LATENT_DIM)

    def test_trains_one_epoch_without_error(self):
        dataset = PacmanDataset(gamestates=make_synthetic_sequences())
        model = MLPAutoencoder(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[16, 8],
            pooling=False,
        )
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

    def test_non_finite_loss_from_the_start_raises(self):
        """NaN on every epoch leaves no checkpoint to fall back on, so training must fail
        loudly rather than run to max_epochs and leave the caller without a _best.pth."""
        data = make_synthetic_sequences()
        data[0, 0, 0] = np.nan  # obs_mask zeroes its loss, but NaN * 0 is still NaN
        model = VanillaVAE(input_dim=N_FEATURES, seq_len=SEQ_LEN, latent_dim=LATENT_DIM, hidden_dims=[8, 16])
        trainer = VAE_Trainer(max_epochs=5, batch_size=N_SAMPLES, validation_split=0, verbose=False)

        with pytest.raises(FloatingPointError):
            trainer.fit(model, PacmanDataset(gamestates=data))

        assert len(trainer.train_loss_list) == 1

    def test_non_finite_loss_after_a_good_epoch_keeps_the_best_checkpoint(self, tmp_path):
        """Divergence after a finite epoch stops training early and leaves that epoch's
        checkpoint on disk for the caller to embed with."""

        class DivergesOnSecondEpoch(VanillaVAE):
            epochs_started = 0

            def train(self, mode: bool = True):
                if mode:
                    self.epochs_started += 1
                return super().train(mode)

            def forward(self, *args, **kwargs):
                recon, mu, log_var = super().forward(*args, **kwargs)
                if self.epochs_started >= 2:
                    recon = recon * float("nan")
                return [recon, mu, log_var]

        model = DivergesOnSecondEpoch(input_dim=N_FEATURES, seq_len=SEQ_LEN, latent_dim=LATENT_DIM, hidden_dims=[8, 16])
        trainer = VAE_Trainer(
            max_epochs=5,
            batch_size=4,
            validation_split=0.3,
            verbose=False,
            save_model=True,
            best_path=str(tmp_path / "best.pth"),
            last_path=str(tmp_path / "last.pth"),
        )

        trainer.fit(model, PacmanDataset(gamestates=make_synthetic_sequences()))

        assert len(trainer.train_loss_list) == 2
        state = torch.load(tmp_path / "best.pth")
        assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())

    def test_decoder_output_is_unbounded(self):
        """The reconstruction must be able to reach any real value, since global
        normalization z-scores most features and ~28% of pacman_attack values exceed +-1.
        Pinning the last conv to a constant 3.0 makes the check exact: with a bounded
        output activation (the image-VAE Tanh this used to end in) it would come out at 0.995."""
        model = VanillaVAE(
            input_dim=N_FEATURES,
            seq_len=SEQ_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[8, 16],
        )
        last_layer = model.final_layer[-1]
        assert isinstance(last_layer, torch.nn.Conv1d)

        with torch.no_grad():
            last_layer.weight.zero_()
            last_layer.bias.fill_(3.0)
            recon = model.decode(torch.randn(N_SAMPLES, LATENT_DIM))

        assert torch.allclose(recon, torch.full_like(recon, 3.0))


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


class TestInfReplacement:
    """A feature that is inf at every real timestep (a ghost that never leaves the house in
    first_5_seconds) must not inherit the -999 padding sentinel as its "maximum"."""

    PADDING = -999.0

    def make_padded_with_all_inf_feature(self):
        data = make_synthetic_sequences()
        data[..., 1] = np.inf  # feature 1 is never finite on a real timestep
        data[:, -3:, :] = self.PADDING  # last 3 timesteps are padding
        return data

    @pytest.mark.parametrize("dataset_cls", [PacmanDataset, ImputationDataset])
    def test_datasets_fill_an_all_inf_feature_with_zero(self, dataset_cls):
        dataset = dataset_cls(gamestates=self.make_padded_with_all_inf_feature(), padding_value=self.PADDING)

        valid = dataset.padding_mask.bool()
        assert torch.isfinite(dataset.gamestates).all()
        assert (dataset.gamestates[valid][:, 1] == 0).all()
        assert (dataset.gamestates[~valid] == 0).all()

    def test_numpy_helper_ignores_padding(self):
        from src.utils import replace_inf_with_feature_max

        cleaned = replace_inf_with_feature_max(self.make_padded_with_all_inf_feature(), padding_value=self.PADDING)

        assert (cleaned[:, :-3, 1] == 0).all()
        assert (cleaned[:, -3:, :] == self.PADDING).all()  # padding itself is left for the caller
