import torch
import torch.nn as nn
import torch.nn.functional as F


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
             x_h: torch.Tensor,
             x: torch.Tensor,
             padding_mask: torch.Tensor | None = None,
             obs_mask: torch.Tensor | None = None,
             loss_mask: torch.Tensor | None = None): 
        
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
            numerator = (recon_loss * padding_mask).sum()
            denominator = combined_mask.sum().clamp_min(1.0)
            recon_loss = numerator / (denominator + 1e-8)
        else:
            # No padding mask, just use combined valid-mask
            combined_mask = torch.ones_like(recon_loss)
            if loss_mask is not None:
                combined_mask = combined_mask * loss_mask
            if obs_mask is not None:
                combined_mask = combined_mask * obs_mask
            numerator = recon_loss.sum()
            denominator = combined_mask.sum().clamp_min(1.0)
            recon_loss = numerator / (denominator + 1e-8)

        return recon_loss
    
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
