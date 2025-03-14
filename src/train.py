import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger  # optional, for logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datamodule.datamodule import TrajectoryDataModule
from src.models.LSTM import LSTMAutoencoder

def train_model():
    # 1. Initialize the DataModule
    data_module = TrajectoryDataModule(
        data_folder='data/',  # Replace with your data path
        batch_size=32,
        series_type=['position'],  # Add other types if needed: ['position', 'movements', 'input']
        include_game_state_vars=False,
        include_timesteps=True,
        max_sequence_length=None  # Set to specific value if needed
    )

    # 2. Prepare and setup the data
    data_module.prepare_data()
    data_module.setup()

    # 3. Initialize the model
    model = LSTMAutoencoder(
        input_dim=data_module.feature_dimension,
        latent_dim=32,  # Adjust latent dimension as needed
        sequence_length=data_module.sequence_length,
        learning_rate=1e-3
    )

    # 4. Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='lstm_autoencoder-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )

    # 5. Optional: Initialize WandB logger
    # wandb_logger = WandbLogger(project="pacman-trajectories", name="lstm-autoencoder")

    # 6. Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        # logger=wandb_logger,  # Remove if not using WandB
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1
        
    )

    # 7. Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train_model()