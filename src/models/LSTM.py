from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple, Any, Optional
from src.models.base import BaseAutoencoder

class LSTMencoder(nn.Module):
    ## Encoder module
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=latent_dim,
                            batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        return hidden[0] 
        
class LSTMdecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size=latent_dim,
                           hidden_size=latent_dim,  # Use same size as latent dimension
                           batch_first=True)
        # Add a linear layer to map from latent_dim to output_dim
        self.output_layer = nn.Linear(latent_dim, output_dim)
        
    def forward(self, z):
        # Repeat latent vector for each timestep
        z_repeated = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # Pass through LSTM
        hidden_states, _ = self.lstm(z_repeated)
        # Map to output dimension
        output = self.output_layer(hidden_states)
        return output


class LSTMAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim: int, latent_dim: int, sequence_length: int, learning_rate: float = 1e-3):
        super().__init__(input_dim, latent_dim, learning_rate)
        self.encoder = LSTMencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = LSTMdecoder(latent_dim=latent_dim, output_dim=input_dim, sequence_length=sequence_length)
        
    def encode(self,x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    # forward def in base.


    
