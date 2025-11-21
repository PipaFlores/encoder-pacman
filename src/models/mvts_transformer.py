from typing import Optional, Any
import math
import os
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from .loss import MaskedMSELoss, l2_reg_loss

## Adapted from https://github.com/gzerveas/mvts_transformer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class MeanPooling(nn.Module):
    """
    Masked mean pooling over the token hidden dimensions

    Input: x -> [batch, seq_length, hidden_dim] : Transformer's output (before linear projection to original feature dimensions)
           mask -> [batch, seq_length] 1 valid tokens, 0 for padding
        
    Output: z -> [batch, hidden_dim] : Sequence-level embedding (one per sample)
    
    """

    def forward(self, x : Tensor, 
                mask : Tensor | None =None):
        if mask is None:
            return x.mean(dim=1) # mean of values over the token dimension

        mask = mask.unsqueeze(-1) # for broadcasting
        x_masked = x * mask # Only non-masked values embeddings
        lengths = mask.sum(dim=1).clamp(min=1e-6) # Length of valid time-steps per dimension, per sample
        return x_masked.sum(dim=1) / lengths # Masked mean
    
    
class SumPooling(nn.Module):
    """
    Masked sum pooling over the token hidden dimensions

    Input: x -> [batch, seq_length, hidden_dim] : Transformer's output (before linear projection to original feature dimensions)
           mask -> [batch, seq_length] 1 valid tokens, 0 for padding
        
    Output: z -> [batch, hidden_dim] : Sequence-level embedding (one per sample)
    
    """

    def forward(self, x: Tensor, 
                mask : Tensor | None = None):

        if mask is None:
            return x.sum(dim=1)
        
        mask = mask.unsqueeze(-1)
        x_masked = x * mask
        return x_masked.sum(dim=1)

class MaxPooling(nn.Module):
    """
    Masked max pooling over the token hidden dimensions

    Input: x -> [batch, seq_length, hidden_dim] : Transformer's output (before linear projection to original feature dimensions)
           mask -> [batch, seq_length] 1 valid tokens, 0 for padding
        
    Output: z -> [batch, hidden_dim] : Sequence-level embedding (one per sample)
    
    """

    def forward(self,
                x: Tensor,
                mask: Tensor | None = None):
        
        if mask is None:
            return x.max(dim=1)

        mask = mask.unsqueeze(-1)
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(mask == 0, neg_inf)

        return x_masked.max(dim=1).values
    
class CLSPooling(nn.Module):
    """
    This follow from BERT's special token `[CLS]` which is usually present
    at the beginning of each sequence (at index 0).

    Input: x -> [batch, seq_length, hidden_dim] : Transformer's output (before linear projection to original feature dimensions)
           mask is not used here. Included to fit with pooling calls.
        
    Output: z -> [batch, hidden_dim] : Sequence-level embedding (one per sample)

    """

    def __init__(self, 
                 cls_index:int = 0):
        super().__init__()
        self.cls_index = cls_index

    def forward(self, 
                x: Tensor, 
                mask = None):

        return x[:, self.cls_index, :]


def get_pooling_class(pool_type:str):

    if pool_type == "mean":
        return MeanPooling()
    if pool_type == "sum":
        return SumPooling()
    if pool_type == "max":
        return MaxPooling()
    if pool_type == "CLS":
        return CLSPooling()

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src

## Model Architecture

class TSTransformerEncoder(nn.Module):
    """
    This is the main class containing the whole model
    TODO:
     - Verify and adapt loss function
    
    """

    def __init__(self, 
                 feat_dim, # Dimensionality of data features
                 max_len: int | None = None, ## If none, use the max length defined in dataclass  TODO implement 
                 d_model: int = 128,  # or 64, 256, 512
                 n_heads: int = 8, # or 16 
                 num_layers: int = 3, # or 1 
                 dim_feedforward: int = 256, # or 512 
                 dropout=0.1,
                 pos_encoding='fixed', 
                 activation='gelu', 
                 norm='BatchNorm', 
                 freeze=False,
                 pooling_method: str = "mean"):
        """
        Filled with default initialization values
        """
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.mask_token = nn.Parameter(torch.randn(1, 1, feat_dim))

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        self.pooling = get_pooling_class(pooling_method)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        elif norm == 'BatchNorm':
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        latent_representation = self.encode(X, padding_masks, pooling=False)
        output = self.dropout1(latent_representation)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output

    def encode(self, 
               X, 
               padding_masks, 
               pooling = False):
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        latent_representation = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        latent_representation = self.act(latent_representation)  # the output transformer encoder/decoder embeddings don't include non-linearity
        latent_representation = latent_representation.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        if pooling:
            latent_representation = self.pooling.forward(latent_representation, mask = padding_masks)

        return latent_representation
    

class Transformer_Trainer():
    def __init__(self, 
                 max_epochs= 50, 
                 batch_size= 32, 
                 validation_split: float | None = 0.3, 
                 use_imputation: bool = True,
                 lr: float = 0.001,
                 global_regularization: bool = False, # default to false
                 l2_reg: float = 0, ## default to 0
                 gradient_clipping = None,
                 seed: int | None = None, 
                 verbose = True,
                 optim_algorithm: str = "Radam", # FIXME Radam
                 save_model = False,
                 best_path = None,
                 last_path = None,
                 wandb_run = None):
        """
        All configuration stuff should be here

        TODO
        - Simplify elements from main.py and runner.py. Remove everything regarding classification or regression.
        We only care about the unsupervised pre-training.

            - Data preprocessing
                -skip all normalization steps as they are handled in the pa pipeline 
            - Optimizer. L2_regularization (on the optimizer?? or is it both in the optimizer AND in the module)
                - It is a choice (global vs output) if global, then l2_reg goes to weight_decay in optimizer
                If output, then l2_reg goes to output layer (current model l2_reg_loss function)
            - Train
                The epoch training loop is in the runner class



        - Start with the TS benchmark dataset
        - Try to keep our current datamodule class instead of using the one in the og repo.
        
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_imputation = use_imputation
        self.global_regularization = global_regularization
        self.l2_reg = l2_reg
        self.lr = lr # learning rate (?)
        self.gradient_clipping = gradient_clipping
        self.seed = seed
        self.verbose = verbose
        self.optim_algorithm = optim_algorithm
        self.save_model = save_model
        self.best_path = best_path
        self.last_path = last_path
        self.wandb_run = wandb_run

    def fit(self, model:TSTransformerEncoder , data: torch.utils.data.Dataset):
        """
        Fits the given model to the provided dataset using the specified training configuration.

        Args:
            model (nn.Module): The PyTorch model to be trained. Must implement a forward method.
            data (torch.utils.data.Dataset): The dataclass with data to train on, and masks (if needed). 
            Should return batches as dicts with "data" keys. If normalized, it should be done before this fit() method.
            i.e., def __getitem__() returns {'data': data, ...} No labels are needed, as this is for Autoencoder training.

        Returns:
            None. Updates the model in-place and stores training/validation loss history in self.train_loss_list and self.val_loss_list.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        ## Initialize Optimizer
        if self.global_regularization:
            weight_decay = self.l2_reg
            output_reg = 0
        else:
            weight_decay = 0
            output_reg = self.l2_reg

        if self.optim_algorithm == "Radam":
            optimizer = torch.optim.RAdam(params=self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif self.optim_algorithm == "Adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer type ({self.optim_algorithm}) not supported")
        
        ## Initialize Loss
        loss = MaskedMSELoss()
        
        ### Data split
        ## FIXME there is a missing collation function used by the og pipeline
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

                noise_mask = None
                if self.use_imputation:
                    noise_mask = batch["noise_mask"].to(self.device)
                    # mask the input: 0 = masked (set to model.mask_token), 1 = keep original
                    mask_token = self.model.mask_token.expand(x.size(0), x.size(1), -1)
                    x_masked = x * noise_mask + (1 - noise_mask) * mask_token
                else:
                    x_masked = x

                # Padded steps are feeded to the transformed so they are ignored in attention
                padding_mask = batch.get("padding_mask", None)
                if padding_mask is not None: 
                    padding_mask = batch["padding_mask"].to(self.device)

                x_h = self.model(X=x_masked, padding_masks=padding_mask.bool())

                # Observation masked loss for missing elements
                # e.g., astar distance = inf when ghosts in house 
                obs_mask = batch.get("obs_mask", None) 
                if obs_mask is not None:
                    obs_mask = batch["obs_mask"].to(self.device)

                # For loss, we want to compute loss of masked positions
                # For this we invert the original noise mask
                # noise_mask (1 = keep, 0 = mask) -> loss_mask (1 = predict (mask), 0 = ignore (unmasked))
                loss_mask = (1 - noise_mask) if noise_mask is not None else None
                batch_loss = loss.forward(x_h, x, padding_mask, obs_mask, loss_mask)

                if not self.global_regularization and output_reg is not None:
                    total_loss = batch_loss + output_reg * l2_reg_loss(self.model) # TODO what is the output layer????
                else:
                    total_loss = batch_loss

                loss_sum += batch_loss.item()

                # Backprop and weight update
                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping in case of exploding gradients
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
                optimizer.step()

            if self.validation_split > 0:
                self.model.eval()
                val_loss_sum = 0 

                for batch in val_iter:
                    with torch.no_grad():
                        x = batch["data"].to(self.device)

                        noise_mask = None
                        if self.use_imputation:
                            noise_mask = batch["noise_mask"].to(self.device)
                            mask_token = self.model.mask_token.expand(x.size(0), x.size(1), -1)
                            x_masked = x * noise_mask + (1 - noise_mask) * mask_token
                        else:
                            x_masked = x
                        
                        padding_mask = batch.get("padding_mask", None)
                        if padding_mask is not None:
                            padding_mask = batch["padding_mask"].to(self.device)

                        x_h = self.model(X=x_masked, padding_masks=padding_mask.bool())
                        
                        obs_mask = batch.get("obs_mask", None) 
                        if obs_mask is not None:
                            obs_mask = batch["obs_mask"].to(self.device)

                        loss_mask = (1 - noise_mask) if noise_mask is not None else None
                        batch_loss = loss.forward(x_h, x, padding_mask, obs_mask,loss_mask)
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
            if self.wandb_run and WANDB_AVAILABLE:
                self.wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss" : epoch_train_loss,
                        "val_loss": epoch_val_loss if self.validation_split > 0 else None,
                    }
                )
        