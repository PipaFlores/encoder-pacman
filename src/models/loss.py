import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedMSELoss(nn.Module):
    """
    Reconstruction MSE loss, with optional masking 
    for handling padded or missing values, and optional deep clustering loss.

    Args:
        x_h (torch.Tensor): The reconstructed output, shape [batch, seq_length, feature_dim].
        x (torch.Tensor): The original input, shape [batch, seq_length, feature_dim].
        padding_mask (torch.Tensor, optional): Sequence mask of shape [batch, seq_length], 
            1 for valid timesteps (not padding), 0 for padding. Default: None.
        obs_mask (torch.Tensor, optional): Observation mask of shape [batch, seq_length, feature_dim], 
            1 for finite values, 0 for missing/infinite. Default: None.
        loss_mask(torch.Tensor, optional): Loss mask of shape [batch, seq_length_feature_dim].
            1 for including loss, 0 to ignore. Default: None.
    Returns:
        torch.Tensor: The total loss value
    """
    def forward(self, 
             x_h: Tensor,
             x: Tensor,
             padding_mask: Tensor | None = None,
             obs_mask: Tensor | None = None,
             loss_mask: Tensor | None = None,
             reduce_to_mean: bool = True): 
        
        recon_loss = nn.functional.mse_loss(x_h, x, reduction="none")

        if loss_mask is not None:
            # loss_mask = 1 predict (compute loss), 0 = ignore
            recon_loss = recon_loss * loss_mask

        if obs_mask is not None:
            recon_loss = recon_loss * obs_mask  # element-wise masking

        if padding_mask is not None:
            # [batch, seq_length] -> [batch, seq_length, 1] (broadcast for feature_dim)
            if padding_mask.ndim == 2 and recon_loss.ndim == 3:
                padding_mask = padding_mask.unsqueeze(-1)
            elif padding_mask.ndim == 1 and recon_loss.ndim == 2:
                padding_mask = padding_mask.unsqueeze(-1)
            # Combine all masks for correct denominator
            combined_mask = torch.ones_like(recon_loss)
            if loss_mask is not None:
                combined_mask = combined_mask * loss_mask
            if obs_mask is not None:
                combined_mask = combined_mask * obs_mask
            combined_mask = combined_mask * padding_mask

            if reduce_to_mean:
                numerator = (recon_loss * padding_mask).sum()
                denominator = combined_mask.sum().clamp_min(1.0)
                recon_loss = numerator / (denominator + 1e-8)
            else:
                # Reduce by sum (no mean): elementwise masked loss, sum over all valid (or keep as is)
                recon_loss = (recon_loss * padding_mask)

            # No padding mask, just use combined valid-mask
            # This is in the 'else' path for padding_mask is None
        else:
            combined_mask = torch.ones_like(recon_loss)
            if loss_mask is not None:
                combined_mask = combined_mask * loss_mask
            if obs_mask is not None:
                combined_mask = combined_mask * obs_mask

            if reduce_to_mean:
                numerator = recon_loss.sum()
                denominator = combined_mask.sum().clamp_min(1.0)
                recon_loss = numerator / (denominator + 1e-8)
            else:
                # Reduce by sum (no mean): elementwise masked loss
                # recon_loss stays un-reduced (optionally apply the combined mask)
                recon_loss = recon_loss

        return recon_loss


class VAELoss(nn.Module):
    
    def __init__(self,
                 kld_weight: float = 0.00025
                 ):
        """
        Loss class for Variational Autoencoder (VAE) combining reconstruction loss (MSE)
        and Kullback-Leibler divergence (KLD).

        Args:
            kld_weight (float): Weight for the KL divergence term relative to the reconstruction loss.
        """
        super(VAELoss).__init__()

        self.kld_weight = kld_weight

    def forward(self,
                recon : Tensor,
                input: Tensor,
                mu: Tensor,
                log_var: Tensor,
                padding_mask: Tensor | None = None,
                obs_mask: Tensor | None = None,
                loss_mask: Tensor | None = None,
                reduce_to_mean: bool = True):
        
        recon_loss = F.mse_loss(recon, input)  # reduced with mean (default)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) , dim = 0)

        total_loss = recon_loss + kld_loss * self.kld_weight

        return total_loss, kld_loss


    # def loss_function(self,
    #                   *args,
    #                   **kwargs) -> dict:
    #     """
    #     Computes the VAE loss function.
    #     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]

    #     kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    #     recons_loss =F.mse_loss(recons, input)


    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     loss = recons_loss + kld_weight * kld_loss
    #     return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}



def l2_reg_loss(model):
        """
        L2 norm of output layer weights.
        For regularization.
        I do not know yet if this is used for the autoregressive task, but it might be useful to add it to the loss.
        So, keeping it here for reference.
        """
        for name, param in model.named_parameters():
            if name == "output_layer.weight":
                return torch.sum(torch.square(param))
