import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseAutoencoder(pl.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

    def encode(self, x):
        """Convert input data into latent representation"""
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(self, z):
        """Convert latent representation back to original space"""
        raise NotImplementedError("Subclasses must implement decode()")

    def forward(self, x):
        # This defines what happens when you do model(x)
        z = self.encode(x)  # First encode the input
        x_hat = self.decode(z)  # Then decode it
        return x_hat, z  # Return both reconstruction and latent representation

    def masked_reconstruction_loss(
        self, x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate reconstruction loss only on valid timesteps"""
        # Calculate MSE for each element
        mse = nn.MSELoss(reduction="none")(x_hat, x)  # (batch, seq_len, features)

        # Apply mask (expand mask to match features dimension)
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        masked_mse = (mse * mask).sum() / (mask.sum() + 1e-8)

        return masked_mse

    def training_step(self, batch, batch_idx):
        # Unpack the batch dictionary
        x = batch["trajectory"]
        mask = batch["mask"]

        # Forward pass
        x_hat, z = self(x)

        # Calculate loss using mask
        loss = self.masked_reconstruction_loss(x_hat, x, mask)

        # Log the loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["trajectory"]
        mask = batch["mask"]
        x_hat, z = self(x)
        val_loss = self.masked_reconstruction_loss(x_hat, x, mask)
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        # Define which optimizer to use
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
