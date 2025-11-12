from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


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
                 freeze=False):
        """
        Filled with default initialization values
        """
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

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

        latent_representation = self.encode(X, padding_masks)
        output = self.dropout1(latent_representation)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output

    def encode(self, X, padding_masks):
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        latent_representation = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        latent_representation = self.act(latent_representation)  # the output transformer encoder/decoder embeddings don't include non-linearity
        latent_representation = latent_representation.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        return latent_representation
    
    def loss(self, 
             y_pred: torch.Tensor, 
             y_true: torch.Tensor,
             mask: torch.Tensor = None,
             obs_mask: torch.Tensor = None):
        
        reduction = "none"

        # NOTE BY OG AUTHOR || for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask 
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        
        total_loss = nn.functional.mse_loss(masked_pred, masked_true) 
        ## Calculating the loss of only unmasked stuff.
        ## BEWARE, these masks are probably not the same as padding masks. It might be related
        ## to the noise-masks for the auto-regressive task of denoising (unsupervised) <- THIS IS WHAT WE ARE INTERESTED ON
        
        return total_loss
    
    def l2_reg_loss(self):
        """
        L2 norm of output layer weights.
        For regularization.
        I do not know yet if this is used for the autoregressive task, but it might be useful to add it to the loss.
        So, keeping it here for reference.
        """

        for name, param in self.named_parameters():
            if name == "output_layer.weight":
                return torch.sum(torch.square(param))


class Transformer_Trainer():
    def __init__(self, 
                 max_epochs= 50, 
                 batch_size= 32, 
                 validation_split: float | None = 0.3, 
                 lr: float = 0.001,
                 l2_reg: bool = False,
                 gradient_clipping = None, 
                 verbose = True,
                 optim_algorithm: torch.optim.Optimizer | None = torch.optim.Adam,
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
        self.lr = lr
        self.gradient_clipping = gradient_clipping
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

        ## TODO include weight decay (l2 norm)
        if not self.optim_algorithm:
            optimizer = (
                self.model.configure_optimizer()
                if hasattr(self.model, "configure_optimizer")
                else torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
            )
        else:
            optimizer = self.optim_algorithm(params=self.model.parameters(), lr=self.lr) 
        
        if hasattr(self.model, "loss") and callable(self.model.loss):
            loss = self.model.loss
        else:
            raise AttributeError("Model does not have a callable 'loss' attribute.")

        if self.validation_split > 0:
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

            if self.validation_split > 0:
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
                        torch.save(model.state_dict(), self.best_path)
                else:
                    if epoch_train_loss < best_loss:
                        best_loss = epoch_train_loss
                        torch.save(model.state_dict(), self.best_path)

                torch.save(model.state_dict(), self.last_path)

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
        
        # if self.wandb_run and WANDB_AVAILABLE:
        #     self.wandb_run.finish() ## run is always managed by external calls, so finish it there, after logging other important things (as plots)


### TRAINING CLASSES

class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        # self.printer = utils.Printer(console=console)

        # self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    # def print_callback(self, i_batch, metrics, prefix=''):

    #     total_batches = len(self.dataloader)

    #     template = "{:5.1f}% | batch: {:9d} of {:9d}"
    #     content = [100 * (i_batch / total_batches), i_batch, total_batches]
    #     for met_name, met_value in metrics.items():
    #         template += "\t|\t{}".format(met_name) + ": {:g}"
    #         content.append(met_value)

    #     dyn_string = template.format(*content)
    #     dyn_string = prefix + dyn_string
    #     # self.printer.print(dyn_string)

class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            # TODO: for debugging
            # input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
            # if not input_ok:
            #     print("Input problem!")
            #     ipdb.set_trace()
            #
            # utils.check_model(self.model, verbose=False, stop_on_error=True)

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

            if keep_all:
                per_batch['target_masks'].append(target_masks.cpu().numpy())
                per_batch['targets'].append(targets.cpu().numpy())
                per_batch['predictions'].append(predictions.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])
                per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics

def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config

class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')

        ## Run from config file
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        ## Run from command-line arguments
        # I/O
        self.parser.add_argument('--output_dir', default='./output',
                                 help='Root output directory. Must exist. Time-stamped directories will be created inside.')
        self.parser.add_argument('--data_dir', default='./data',
                                 help='Data directory')
        self.parser.add_argument('--load_model',
                                 help='Path to pre-trained model.')
        self.parser.add_argument('--resume', action='store_true',
                                 help='If set, will load `starting_epoch` and state of optimizer, besides model weights.')
        self.parser.add_argument('--change_output', action='store_true',
                                 help='Whether the loaded model will be fine-tuned on a different task (necessitating a different output layer)')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='If set, will save model weights (and optimizer state) for every epoch; otherwise just latest')
        self.parser.add_argument('--name', dest='experiment_name', default='',
                                 help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
        self.parser.add_argument('--comment', type=str, default='', help='A comment/description of the experiment')
        self.parser.add_argument('--no_timestamp', action='store_true',
                                 help='If set, a timestamp will not be appended to the output directory name')
        self.parser.add_argument('--records_file', default='./records.xls',
                                 help='Excel file keeping all records of experiments')
        # System
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output; otherwise for file")
        self.parser.add_argument('--print_interval', type=int, default=1,
                                 help='Print batch info every this many batches')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')
        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading/preprocessing. By default, equals num. of available cores.')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed',
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')
        # Dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
                                      "If in [0,1], it will be interpreted as a proportion of the dataset, "
                                      "otherwise as an integer absolute number of samples")
        self.parser.add_argument('--test_only', choices={'testset', 'fold_transduction'},
                                 help='If set, no training will take place; instead, trained model will be loaded and evaluated on test set')
        self.parser.add_argument('--data_class', type=str, default='weld',
                                 help="Which type of data should be processed.")
        self.parser.add_argument('--labels', type=str,
                                 help="In case a dataset contains several labels (multi-task), "
                                      "which type of labels should be used in regression or classification, i.e. name of column(s).")
        self.parser.add_argument('--test_from',
                                 help='If given, will read test IDs from specified text file containing sample IDs one in each row')
        self.parser.add_argument('--test_ratio', type=float, default=0,
                                 help="Set aside this proportion of the dataset as a test set")
        self.parser.add_argument('--val_ratio', type=float, default=0.2,
                                 help="Proportion of the dataset to be used as a validation set")
        self.parser.add_argument('--pattern', type=str,
                                 help='Regex pattern used to select files contained in `data_dir`. If None, all data will be used.')
        self.parser.add_argument('--val_pattern', type=str,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the validation set.
                            If None, a positive `val_ratio` will be used to reserve part of the common data set.""")
        self.parser.add_argument('--test_pattern', type=str,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the test set.
                            If None, `test_ratio`, if specified, will be used to reserve part of the common data set.""")
        self.parser.add_argument('--normalization',
                                 choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
                                 default='standardization',
                                 help='If specified, will apply normalization on the input features of a dataset.')
        self.parser.add_argument('--norm_from',
                                 help="""If given, will read normalization values (e.g. mean, std, min, max) from specified pickle file.
                            The columns correspond to features, rows correspond to mean, std or min, max.""")
        self.parser.add_argument('--subsample_factor', type=int,
                                 help='Sub-sampling factor used for long sequences: keep every kth sample')
        # Training process
        self.parser.add_argument('--task', choices={"imputation", "transduction", "classification", "regression"},
                                 default="imputation",
                                 help=("Training objective/task: imputation of masked values,\n"
                                       "                          transduction of features to other features,\n"
                                       "                          classification of entire time series,\n"
                                       "                          regression of scalar(s) for entire time series"))
        self.parser.add_argument('--masking_ratio', type=float, default=0.15,
                                 help='Imputation: mask this proportion of each variable')
        self.parser.add_argument('--mean_mask_length', type=float, default=3,
                                 help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.")
        self.parser.add_argument('--mask_mode', choices={'separate', 'concurrent'}, default='separate',
                                 help=("Imputation: whether each variable should be masked separately "
                                       "or all variables at a certain positions should be masked concurrently"))
        self.parser.add_argument('--mask_distribution', choices={'geometric', 'bernoulli'}, default='geometric',
                                 help=("Imputation: whether each mask sequence element is sampled independently at random"
                                       "or whether sampling follows a markov chain (stateful), resulting in "
                                       "geometric distributions of masked squences of a desired mean_mask_length"))
        self.parser.add_argument('--exclude_feats', type=str, default=None,
                                 help='Imputation: Comma separated string of indices corresponding to features to be excluded from masking')
        self.parser.add_argument('--mask_feats', type=str, default='0, 1',
                                 help='Transduction: Comma separated string of indices corresponding to features to be masked')
        self.parser.add_argument('--start_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the beginning of time series which will not be masked')
        self.parser.add_argument('--end_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the end of time series which will not be masked')
        self.parser.add_argument('--harden', action='store_true',
                                 help='Makes training objective progressively harder, by masking more of the input')

        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='Number of training epochs')
        self.parser.add_argument('--val_interval', type=int, default=2,
                                 help='Evaluate on validation set every this many epochs. Must be >= 1.')
        self.parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate (default holds for batch size 64)')
        self.parser.add_argument('--lr_step', type=str, default='1000000',
                                 help='Comma separated string of epochs when to reduce learning rate by a factor of 10.'
                                      ' The default is a large value, meaning that the learning rate will not change.')
        self.parser.add_argument('--lr_factor', type=str, default='0.1',
                                 help=("Comma separated string of multiplicative factors to be applied to lr "
                                       "at corresponding steps specified in `lr_step`. If a single value is provided, "
                                       "it will be replicated to match the number of steps in `lr_step`."))
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Training batch size')
        self.parser.add_argument('--l2_reg', type=float, default=0,
                                 help='L2 weight regularization parameter')
        self.parser.add_argument('--global_reg', action='store_true',
                                 help='If set, L2 regularization will be applied to all weights instead of only the output layer')
        self.parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                                 help='Metric used for defining best epoch')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer')

        # Model
        self.parser.add_argument('--model', choices={"transformer", "LINEAR"}, default="transformer",
                                 help="Model class")
        self.parser.add_argument('--max_seq_len', type=int,
                                 help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""")
        self.parser.add_argument('--data_window_len', type=int,
                                 help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length 
                                 (size of transformer layers).""")
        self.parser.add_argument('--d_model', type=int, default=64,
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimension of dense feedforward part of transformer layer')
        self.parser.add_argument('--num_heads', type=int, default=8,
                                 help='Number of multi-headed attention heads')
        self.parser.add_argument('--num_layers', type=int, default=3,
                                 help='Number of transformer encoder layers (blocks)')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout applied to most transformer encoder layers')
        self.parser.add_argument('--pos_encoding', choices={'fixed', 'learnable'}, default='fixed',
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                                 help='Activation to be used in transformer encoder')
        self.parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                                 help='Normalization layer to be used internally in transformer encoder')

    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(',')]
        args.lr_factor = [float(i) for i in args.lr_factor.split(',')]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor), "You must specify as many values in `lr_step` as in `lr_factors`"

        if args.exclude_feats is not None:
            args.exclude_feats = [int(i) for i in args.exclude_feats.split(',')]
        args.mask_feats = [int(i) for i in args.mask_feats.split(',')]

        if args.val_pattern is not None:
            args.val_ratio = 0
            args.test_ratio = 0

        return args