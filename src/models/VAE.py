import torch
import math
import os
from torch import nn, Tensor
from torch.nn import functional as F


class VanillaVAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 seq_len : int,
                 latent_dim: int = 128,
                 hidden_dims: list | None = None,
                 **kwargs) -> None:
        
        super(VanillaVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            input_dim = h_dim # do not modify the self.input_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]* seq_len, latent_dim) # Linear proj to gaussian mean
        self.fc_var = nn.Linear(hidden_dims[-1]* seq_len, latent_dim) # Same to variance


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * seq_len)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride =1,
                                       padding=1
                                       ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1
                                               ),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= self.input_dim,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        # self.final_layer = nn.Sequential(
        #     nn.Linear(hidden_dims[-1],
        #               )
        # )

    def encode(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [batch, seq_len, features]
        :return: (Tensor) List of latent codes
        """
        input = input.permute(0,2,1) # Conv layers expect [batch, feats, seq_len]

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder[-1][0].out_channels, self.seq_len)
        result = self.decoder(result)
        result = self.final_layer(result)

        result = result.permute(0,2,1) # return in standard shape of [batch, seq_len, features]

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> list[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    

class VAE_Trainer():
    def __init__(self, 
                 max_epochs= 50, 
                 batch_size= 32, 
                 validation_split: float | None = 0.3, 
                 learning_rate: float = 0.005,
                 kld_weight: float = 0.00025,
                 gradient_clipping = None, 
                 verbose = True,
                 optim_algorithm: torch.optim.Optimizer | None = None,
                 save_model = False,
                 best_path = None,
                 last_path = None,
                 wandb_run = None):
        """
        Trainer class for handling the training loop of the Variational AutoEncoder (VAE).

        Args:
            max_epochs (int): Maximum number of training epochs.
            batch_size (int): Number of samples per batch.
            validation_split (float or None): Fraction of data to use as validation set split. If None, no validation is performed.
            learning_rate (float): Learning rate for the optimization algorithm
            kld_weight (float): weight for the KL divergence loss component
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
        self.learning_rate = learning_rate
        self.kld_weight = kld_weight
        self.verbose = verbose
        self.optim_algorithm = optim_algorithm
        self.save_model = save_model
        self.best_path = best_path
        self.last_path = last_path
        self.wandb_run = wandb_run
        
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
                else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # maybe add weight decay
            )
        else:
            optimizer = self.optim_algorithm(params=self.model.parameters(), lr=self.learning_rate)
        
        from .loss import VAELoss
        loss = VAELoss(kld_weight=self.kld_weight)


        if self.validation_split > 0:
            train_set, val_set = torch.utils.data.random_split(data, [1 - self.validation_split, self.validation_split])
            val_iter = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
        else:
            train_set = data
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        self.train_loss_list, self.val_loss_list = [], []
        best_loss = math.inf

        for epoch in range(self.max_epochs):
            self.model.train()
            loss_sum = 0

            for batch in train_iter:
                x = batch["data"].to(self.device)
                x_h, mu, log_var = self.model(x)

                # masked loss for variable seq_lengths
                padding_mask = batch.get("padding_mask", None)
                if padding_mask is not None: 
                    padding_mask = batch["padding_mask"].to(self.device)

                # Observation masked loss for missing elements
                # e.g., astar distance = inf when ghosts in house 
                obs_mask = batch.get("obs_mask", None) 
                if obs_mask is not None:
                    obs_mask = batch["obs_mask"].to(self.device)

                optimizer.zero_grad()
                batch_loss, kld_loss = loss.forward(recon=x_h,
                                          input=x,
                                          mu=mu,
                                          log_var=log_var,
                                          padding_mask=padding_mask, 
                                          obs_mask=obs_mask) 
                loss_sum += batch_loss.item()
                
                batch_loss.backward()
                # Gradient clipping in case of exploding gradients
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                optimizer.step()

            if self.validation_split > 0:
                self.model.eval()
                val_loss_sum = 0 

                for batch in val_iter:
                    with torch.no_grad():
                        x = batch["data"].to(self.device)
                        x_h, mu, log_var = self.model(x)
                        padding_mask = batch.get("padding_mask", None)
                        if padding_mask is not None:
                            padding_mask = batch["padding_mask"].to(self.device)

                        obs_mask = batch.get("obs_mask", None) 
                        if obs_mask is not None:
                            obs_mask = batch["obs_mask"].to(self.device)

                        batch_loss, kls_loss = loss.forward(recon=x_h,
                                                  input=x,
                                                  mu=mu,
                                                  log_var=log_var,
                                                  padding_mask=padding_mask, 
                                                  obs_mask=obs_mask)
                        val_loss_sum += batch_loss.item()


            epoch_train_loss = loss_sum / len(train_iter)
            self.train_loss_list.append(epoch_train_loss)
            self.model.loss_history = self.train_loss_list

            if self.validation_split > 0:
                epoch_val_loss = val_loss_sum / len(val_iter)
                self.val_loss_list.append(epoch_val_loss)
                self.model.val_loss_history = self.val_loss_list

            if self.save_model:
                if self.best_path is None or self.last_path is None:
                    raise ValueError("save_model=True but best_path or last_path not provided")
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(self.best_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.last_path), exist_ok=True)
                
                if self.validation_split > 0:
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        torch.save(self.model.state_dict(), self.best_path)
                else:
                    if epoch_train_loss < best_loss:
                        best_loss = epoch_train_loss
                        torch.save(self.model.state_dict(), self.best_path)

                torch.save(self.model.state_dict(), self.last_path)

            if self.verbose:
                print(f"Epoch {epoch + 1}: Train loss={epoch_train_loss}, Val loss={epoch_val_loss if self.validation_split > 0 else ''}")
            # if self.wandb_run and WANDB_AVAILABLE:
            #     self.wandb_run.log(
            #         {
            #             "epoch": epoch + 1,
            #             "train_loss" : epoch_train_loss,
            #             "val_loss": epoch_val_loss if self.validation_split > 0 else None,
            #         }
            #     )
        

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
                