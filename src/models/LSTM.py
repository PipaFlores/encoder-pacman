import torch
import torch.nn as nn
import math
import os
from typing import Callable

## LSTM - AutoEncoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, seq_length=None):
        """
        Encoder LSTM network. It is a pretty standard LSTM implementation. 
        Only difference is that instead of outputting a next step prediction"""
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

    def forward(self, X: torch.Tensor):
        out, (h, c) = self.lstm(X) # output hidden_state vector for each timestep, (last h, last c)
        x_encoding = h.squeeze(dim=0) #  [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        ## Implementation from Matan Levi repeats the hidden vector to the seq_length here, but I will do it at the AE forward step.
        return x_encoding, out, (h, c)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout= 0, seq_length=None, last_act = None, forced_teacher = False):
        """
        Decoder LSTM

        Args:
            input_size: number of dimensions of a timestep. If teacher forcing, is equal to data's dimensionality 
            and encoder's input size. Otherwise, it is the hidden_size (if feeding the repeated encoded data)
            hidden_size: size of hidden states (equal to encoder hidden_size).
            output_size: dimensionality of original data, to project the reconstruction from the hidden_size

        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.last_act = last_act

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        self.fully_connected = nn.Linear(in_features=hidden_size, out_features = output_size)
    
    def forward(self, z: torch.Tensor, HC: tuple[torch.Tensor, torch.Tensor]):

        dec_output, (h, c) = self.lstm(z, HC) 

        if self.last_act:
            reconstruction = self.last_act(self.fully_connected(dec_output))
        else:
            reconstruction = self.fully_connected(dec_output)
        
        return reconstruction

class AELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout= 0,last_act = None ,seq_length=None, forced_teacher = False):
        """
        FIXME : Implement forced_teacher
        LSTM Autoencoder (AELSTM) module.

        Args:
            input_size (int): Number of features (channels) in the input time series.
            hidden_size (int): Number of hidden units in the LSTM encoder/decoder.
            dropout (float, optional): Dropout rate for LSTM layers. Default is 0.
            last_act (callable, optional): Activation function to apply to the decoder output. Default is None.
            seq_length (int, optional): Length of the input sequences. Default is None.
            forced_teacher (bool, optional): If True, use teacher forcing in the decoder (feed ground truth as input). Default is False.

        The AELSTM consists of an LSTM encoder and an LSTM decoder. The encoder compresses the input sequence into a latent representation.
        The decoder reconstructs the input sequence from the latent representation. If `forced_teacher` is True, the decoder receives the original
        input at each time step (teacher forcing); otherwise, it receives the repeated encoded vector.

        """
        super(AELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout, seq_length=seq_length)

        if forced_teacher:
            self.decoder = Decoder(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   output_size=input_size, 
                                   dropout=dropout, 
                                   seq_length=seq_length)
        else:
            self.decoder = Decoder(input_size=hidden_size, 
                                   hidden_size=hidden_size, 
                                   output_size=input_size, 
                                   dropout=dropout, 
                                   last_act=last_act,
                                   seq_length=seq_length, 
                                   forced_teacher=forced_teacher)
        
        
    def forward(self, X: torch.Tensor, return_encoding= False):

        x_encoding, _, (h, c)= self.encoder(X)
        z = x_encoding.unsqueeze(1).repeat(1, X.shape[1] , 1) # [batch_dize, hidden_size] -> [batch_size, seq_length, hidden_size]
        reconstruction = self.decoder(z = z, HC = [h, c] )

        if return_encoding:
            return reconstruction, x_encoding
        return reconstruction
    
    def encode(self, X: torch.Tensor):
        x_encoding, _, (__, ___)= self.encoder(X)

        return x_encoding

    
    def configure_optimizers(self):
        """
        Optimization Algorithm.
        """
        return torch.optim.Adam(self.parameters(), lr= 0.001)
    
    def loss(self, 
             x_h: torch.Tensor,
             x: torch.Tensor, 
             mask: torch.Tensor | None = None,
             obs_mask: torch.Tensor | None = None):
        """
        Reconstruction MSE loss
        """

        loss = nn.functional.mse_loss(x_h, x, reduction="none")

        # Handle both obs_mask (element-wise) and mask (sequence length) correctly
        if obs_mask is not None:
            loss = loss * obs_mask  # element-wise masking

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, seq_length] -> [batch, seq_length, 1]
            if obs_mask is not None:
                # Combine both masks for denominator
                combined_mask = (obs_mask * mask)
                loss = (loss * mask).sum() / (combined_mask.sum() + 1e-8)
            else:
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            if obs_mask is not None:
                denominator = obs_mask.sum().clamp_min(1.0)
                loss = loss.sum() / denominator
            else:
                loss = loss.mean()

        return loss

class AE_Trainer():
    def __init__(self, 
                 max_epochs= 50, 
                 batch_size= 32, 
                 validation_split: float | None = 0.3, 
                 gradient_clipping = None, 
                 verbose = True,
                 optim_algorithm: torch.optim.Optimizer | None = None,
                 save_model = False,
                 best_path = None,
                 last_path = None):
        """
        Trainer class for handling the training loop of a PyTorch model Auto-Encoder (reconstruction target).

        Args:
            max_epochs (int): Maximum number of training epochs.
            batch_size (int): Number of samples per batch.
            validation_split (float or None): Fraction of data to use as validation set split. If None, no validation is performed.
            gradient_clipping (float or None): Maximum norm for gradient clipping. If None, no clipping is applied.
            verbose (bool): Whether to print training progress.
            optim_algorithm (torch.optim.Optimizer or None): Optimization algorithm to be used. If None, will use model's configure_optimizer() method or default to Adam.
            save_model (bool): Whether to save the best and last model checkpoints during training.
            best_path (str or None): File path to save the best model checkpoint, including .pth suffix. Required if save_model is True.
            last_path (str or None): File path to save the last model checkpoint, including .pth suffix. Required if save_model is True.
            
        Attributes:
            train_loss_list (list): List storing training loss values for each epoch.
            val_loss_list (list): List storing validation loss values for each epoch.
            device (torch.device): Device (CPU/GPU) used for training.
            model (nn.Module): The trained model instance.
            
        Methods:
            fit(model, data): Trains the model on the provided dataset.
            plot_loss(save_path): Plots and saves training/validation loss curves.
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        self.optim_algorithm = optim_algorithm
        self.save_model = save_model
        self.best_path = best_path
        self.last_path = last_path
        
    def fit(self, model:nn.Module , data: torch.utils.data.Dataset):
        """
        Fits the given model to the provided dataset using the specified training configuration.

        Args:
            model (nn.Module): The PyTorch model to be trained. Must implement a forward method.
            data (torch.utils.data.Dataset): The dataset to train on. Should return batches as dicts with "data" keys.
            i.e., def __getitem__() returns {'data': data, ...} No labels are needed, as this is for Autoencoder training.

        Returns:
            None. Updates the model in-place and stores training/validation loss history in self.train_loss_list and self.val_loss_list.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if not self.optim_algorithm:
            optimizer = (
                self.model.configure_optimizer()
                if hasattr(self.model, "configure_optimizer")
                else torch.optim.Adam(self.model.parameters(), lr=0.001)
            )
        else:
            optimizer = self.optim_algorithm(params=self.model.parameters(), lr=0.001)
        
        loss = self.model.loss if hasattr(self.model, "loss") and callable(self.model.loss) else lambda x_h, x: nn.MSELoss(reduction="sum")(x_h, x)

        if self.validation_split:
            train_set, val_set = torch.utils.data.random_split(data, [1 - self.validation_split, self.validation_split])
            val_iter = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        else:
            train_set = data
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=False)

        self.train_loss_list, self.val_loss_list = [], []
        best_loss = math.inf

        for epoch in range(self.max_epochs):
            self.model.train()
            loss_sum = 0

            for batch in train_iter:
                x = batch["data"].to(self.device)
                x_h = self.model(x)

                # masked loss for variable seq_lengths
                mask = batch.get("mask", None)
                if mask is not None: 
                    mask = batch["mask"].to(self.device)

                # Observation masked loss for missing elements
                # e.g., astar distance = inf when ghosts in house 
                obs_mask = batch.get("obs_mask", None) 
                if obs_mask is not None:
                    obs_mask = batch["obs_mask"].to(self.device)

                optimizer.zero_grad()
                batch_loss = loss(x_h, x, mask, obs_mask)
                loss_sum += batch_loss.item()
                
                batch_loss.backward()
                # Gradient clipping in case of exploding gradients
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                optimizer.step()

            if self.validation_split:
                self.model.eval()
                val_loss_sum = 0 

                for batch in val_iter:
                    with torch.no_grad():
                        x = batch["data"].to(self.device)
                        x_h = self.model(x)
                        mask = batch.get("mask", None)
                        if mask is not None:
                            mask = batch["mask"].to(self.device)

                        obs_mask = batch.get("obs_mask", None) 
                        if obs_mask is not None:
                            obs_mask = batch["obs_mask"].to(self.device)

                        batch_loss = loss(x_h, x, mask, obs_mask)
                        val_loss_sum += batch_loss.item()


            epoch_train_loss = loss_sum / len(train_iter)
            self.train_loss_list.append(epoch_train_loss)
            self.model.loss_history = self.train_loss_list

            if self.validation_split:
                epoch_val_loss = val_loss_sum / len(val_iter)
                self.val_loss_list.append(epoch_val_loss)
                self.model.val_loss_history = self.val_loss_list

            if self.save_model:
                if self.best_path is None or self.last_path is None:
                    raise ValueError("save_model=True but best_path or last_path not provided")
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(self.best_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.last_path), exist_ok=True)
                
                if self.validation_split:
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        torch.save(model.state_dict(), self.best_path)
                else:
                    if epoch_train_loss < best_loss:
                        best_loss = epoch_train_loss
                        torch.save(model.state_dict(), self.best_path)

                torch.save(model.state_dict(), self.last_path)

            if self.verbose:
                print(f"Epoch {epoch + 1}: Train loss={epoch_train_loss}, Val loss={epoch_val_loss if self.validation_split else ''}")

    def plot_loss(self, save_path: str | None = None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(range(len(self.train_loss_list)), self.train_loss_list, label="Train Loss")
        if self.val_loss_list:
            ax.plot(range(len(self.val_loss_list)), self.val_loss_list, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()

        if save_path:
            fig.savefig(save_path, format="png")
        else:
            plt.show()
                
