import torch
from torch import nn, Tensor


class MLPAutoencoder(nn.Module):
    """
    Simple MLP autoencoder, meant as a fast, low-complexity baseline embedder
    to compare against the recurrent/convolutional/transformer/VAE models.

    Unlike VanillaVAE/TimeVAE this is a plain (deterministic) autoencoder, not
    variational - matching AELSTM's interface (`forward`/`encode`, trained with
    `AE_Trainer`) rather than the VAE family's.

    `latent_dim` is a free choice here: a small value (e.g. 2-5) can be used
    directly as the final embedding, skipping the reducer (UMAP/PaCMAP) step in
    PatternAnalysis; a larger value (e.g. 64-128) behaves like the other deep
    embedders and is expected to be reduced by PatternAnalysis before
    clustering.

    Args:
        input_dim (int): Number of input features per timestep.
        seq_len (int): Length of input sequences.
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to 128.
        hidden_dims (list, optional): Sizes of the encoder's hidden layers (mirrored
            in reverse, decoder-side). Defaults to [256, 128].
        dropout (float, optional): Dropout probability applied after each hidden layer.
            Defaults to 0.0.
        pooling (bool, optional): If False (default), matches VanillaVAE's non-pooling
            reference architecture: flatten the whole [seq_len, input_dim] sequence and
            encode/decode it with plain Linear layers. Correct and fully expressive when
            every sample truly has `seq_len` valid timesteps.
            If True, the encoder is applied per-timestep with weights shared across time
            (equivalent to a Conv1d with kernel_size=1), then masked-mean-pooled over
            time (using `lengths`, see `encode`) before the latent projection - needed
            whenever samples are padded to `seq_len`, so that the padding sentinel never
            leaks into the latent space through a fixed per-position weight. Mirrors
            VanillaVAE/TimeVAE's `pooling` argument.
    """

    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 latent_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.0,
                 pooling: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.pooling = pooling

        def mlp_stack(in_dim: int, dims: list) -> tuple[nn.Sequential, int]:
            layers = []
            for h in dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.LeakyReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h
            return nn.Sequential(*layers), in_dim

        # Encoder: per-timestep (pooling=True) or over the full flattened sequence.
        # nn.Linear broadcasts over leading dims, so the same per-timestep stack applies
        # identically to every timestep when fed a [batch, seq_len, input_dim] tensor.
        enc_in_dim = input_dim if pooling else seq_len * input_dim
        self.encoder, enc_out_dim = mlp_stack(enc_in_dim, hidden_dims)
        self.fc_latent = nn.Linear(enc_out_dim, latent_dim)

        # Decoder: always expands the latent into a full per-position [seq_len, hidden]
        # block via one Linear layer (distinct weights per position, like VanillaVAE's
        # decoder_input), then refines it with a per-timestep-shared Linear stack. This
        # avoids collapsing to a single time-invariant output regardless of `pooling`.
        decoder_hidden_dims = list(reversed(hidden_dims))
        self.decoder_input = nn.Linear(latent_dim, decoder_hidden_dims[0] * seq_len)
        self.decoder, dec_out_dim = mlp_stack(decoder_hidden_dims[0], decoder_hidden_dims[1:])
        self.final_layer = nn.Linear(dec_out_dim, input_dim)

    def encode(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Encodes the input into the latent space.

        :param x: (Tensor) Input tensor [batch, seq_len, input_dim].
        :param lengths: (Tensor, optional) [batch], number of valid (non-padded)
            timesteps per sample. Only used when pooling=True (see __init__); ignored
            otherwise.
        :return: (Tensor) Latent codes [batch, latent_dim].
        """
        if self.pooling:
            hidden = self.encoder(x)  # [batch, seq_len, hidden_dims[-1]]
            if lengths is not None:
                mask = (
                    torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
                ).unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = hidden.mean(dim=1)
        else:
            pooled = self.encoder(x.reshape(x.shape[0], -1))

        return self.fc_latent(pooled)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the latent codes back onto the sequence space.

        :param z: (Tensor) [batch, latent_dim]
        :return: (Tensor) Reconstruction [batch, seq_len, input_dim]
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], self.seq_len, -1)
        result = self.decoder(result)
        return self.final_layer(result)

    def forward(self, x: Tensor, lengths: Tensor | None = None, return_encoding: bool = False):
        """
        Runs the encoder-decoder pass.

        :param x: (Tensor) Input tensor [batch, seq_len, input_dim].
        :param lengths: (Tensor, optional) [batch], see `encode`.
        :param return_encoding: (bool) If True, also return the latent codes.
        :return: reconstruction, or (reconstruction, latent codes) if return_encoding.
        """
        z = self.encode(x, lengths=lengths)
        recon = self.decode(z)

        if return_encoding:
            return recon, z
        return recon
