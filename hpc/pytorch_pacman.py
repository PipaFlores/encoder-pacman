
import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
from umap import UMAP
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import AE_Trainer, AELSTM
from src.datahandlers import PacmanDataset, PacmanDataReader, Trajectory

def slice_seq_of_each_level(
        reader: PacmanDataReader,
        start_step=0,
        end_step=-1,
        FEATURES= [
            # "score", 
            # "lives", 
            # "pacman_attack",
            "Pacman_X",
            "Pacman_Y",
            # "Ghost1_X",
            # "Ghost1_Y",            
            # "Ghost2_X",
            # "Ghost2_Y",
            # "Ghost3_X",
            # "Ghost3_Y",
            # "Ghost4_X",
            # "Ghost4_Y",
            ]
    )-> tuple[list[np.ndarray], list[dict], list[Trajectory]]:
    """
    Get a slice of each level from the start gamestep until the end_gamestep.
    If end_step is -1, get the whole level play.

    returns (list[sequences], list[dict], list[traj])
    """
    sequence_list = []
    metadata_list = []
    traj_list = []
    for level_id in reader.level_df["level_id"].unique():
        gamestates, metadata = reader._filter_gamestate_data(level_id=level_id)
        gamestates = gamestates[FEATURES]

        gamestates = gamestates.iloc[start_step:end_step]
        traj = reader.get_partial_trajectory(level_id=level_id, start_step=start_step, end_step=end_step)

        sequence_list.append(gamestates.to_numpy())
        metadata_list.append(metadata)
        traj_list.append(traj)

    return sequence_list, metadata_list, traj_list


def padding_sequences(sequence_list:list[np.ndarray], 
                      padding_value = -999.0):

    # Find the maximum sequence length
    max_seq_length = max(x.shape[0] for x in sequence_list)
    n = len(sequence_list)
    n_features = sequence_list[0].shape[1]

    # Pad sequences with np.nan (or 0, or any value) to max_seq_length
    X_padded = np.full((n, max_seq_length, n_features), padding_value , dtype=np.float32)
    for i, x in enumerate(sequence_list):
        seq_len = x.shape[0]
        X_padded[i, :seq_len, :] = x

    return X_padded

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize autoencoder models with specified parameters.")
    parser.add_argument('--n-epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--latent-space', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--validation-split', type=float, default=0.3, help="Fraction of data to be used as validation set")
    parser.add_argument('--features', type=str, default= "Pacman", help="Which combination of features to use")
    # parser.add_argument('--input_size', type=int, default=2, help='Input size (number of channels/dimensions)')
    parser.add_argument('--verbose', action='store_true', help='Verbosity flag')
    parser.add_argument('--sequence-type', type= str, default="", help= "On what type of sequences to train the model, see source code")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### SLICE DATA
    reader = PacmanDataReader(data_folder="../data", verbose=args.verbose)

    # Check which features to use based on args.features
    if args.features == "Pacman":
        FEATURES = [
            "Pacman_X",
            "Pacman_Y",
        ]
    elif args.features == "Pacman_Ghosts":
        FEATURES = [
            "Pacman_X",
            "Pacman_Y",
            "Ghost1_X",
            "Ghost1_Y",
            "Ghost2_X",
            "Ghost2_Y",
            "Ghost3_X",
            "Ghost3_Y",
            "Ghost4_X",
            "Ghost4_Y",
        ]
    else:
        raise ValueError(f"Unknown features selection: {args.features}")

    N_FEATURES = len(FEATURES)

    SEQUENCE_TYPE = args.sequence_type

    if SEQUENCE_TYPE == "": ## hard-coded type for non-bash based iterations
        SEQUENCE_TYPE = "first_5_seconds"

    if SEQUENCE_TYPE == "first_5_seconds":
        _, sequence_list, _, _ = reader.slice_seq_of_each_level(
            start_step=0, end_step=100, FEATURES=FEATURES, make_gif=False
        )
    elif SEQUENCE_TYPE == "whole_level":
        _, sequence_list, _, _ = reader.slice_seq_of_each_level(
            start_step=0, end_step=-1, FEATURES=FEATURES, make_gif=False
        )
    elif SEQUENCE_TYPE == "last_5_seconds":
        _, sequence_list, _, _ = reader.slice_seq_of_each_level(
            start_step=-100, end_step=-1, FEATURES=FEATURES, make_gif=False
        )
    else:
        raise ValueError(f"Sequence type ({SEQUENCE_TYPE}) not valid")

    X_padded = padding_sequences(sequence_list=sequence_list)

    data_tensor = PacmanDataset(X_padded)
    data_tensor[:]["data"].to(device)

    print(f"loaded data tensor of shape {data_tensor.gamestates.shape}")

    ### TRAIN MODEL
    model_path = os.path.join(
        "trained_models",
        SEQUENCE_TYPE,
        "f" + str(N_FEATURES)
    )

    trainer = AE_Trainer(max_epochs=args.n_epochs, 
                        batch_size=32, 
                        validation_split=args.validation_split,
                        save_model=True,
                        best_path=os.path.join(model_path, f"AELSTM_h{args.latent_space}_e{args.n_epochs}_best.pth"),
                        last_path=os.path.join(model_path, f"AELSTM_h{args.latent_space}_e{args.n_epochs}_last.pth"))

    autoencoder = AELSTM(input_size=data_tensor[0]["data"].shape[1], hidden_size=args.latent_space)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(
        os.path.join(
            "trained_models",
            "loss_plots",
            SEQUENCE_TYPE,
            "f" + str(N_FEATURES)
            ), 
        exist_ok=True
        )
    os.makedirs(
        os.path.join(
            "trained_models",
            "latent_space_plots",
            SEQUENCE_TYPE,
            "f" + str(N_FEATURES)
            ), 
        exist_ok=True
        )


    trainer.fit(autoencoder, data_tensor)
    print(f"Model saved to {model_path}")

    trainer.plot_loss(
        os.path.join(  
            "trained_models",
            "loss_plots",
            SEQUENCE_TYPE,
            "f" + str(N_FEATURES),
            f"{autoencoder.__class__.__name__}_h{args.latent_space}_e{args.n_epochs}.png"
            )
        )

    ### Batch EMBEDD

    BATCH_SIZE = 32  # Adjust based on your available memory
    all_embeddings = [] # empty list to be filled during batch processing.

    autoencoder.eval()
    with torch.no_grad():  # Important: disable gradients for inference
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch_data = data_tensor[i:i+BATCH_SIZE]["data"].to(device)
            
            # Get embeddings for this batch
            _, batch_embeddings = autoencoder.forward(batch_data, return_encoding=True)
            all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory

    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)

    ## reduce
    reducer = UMAP()
    embeddings_2D = reducer.fit_transform(embeddings.detach().numpy())


    ### CLUSTER

    clusterers = {}
    clusterers["HDBSCAN"] = HDBSCAN(min_cluster_size=20)
    
    # N_CLUSTERS = 6 ## for k-means
    # clusterers[f"Kmeans (k={N_CLUSTERS})"] = KMeans(n_clusters=6)

    labels = {}
    for name, clusterer in clusterers.items():
        labels[name] = clusterer.fit_predict(embeddings_2D)

    ## VISUALIZE
    fig, axs = plt.subplots(1, len(labels.values()), figsize=(6 * len(labels.values()), 6))

    # Ensure axs is always a list
    if len(labels.values()) == 1:
        axs = [axs]

    for i, (name, predictions) in enumerate(labels.items()):
        axs[i].scatter(embeddings_2D[:,0], embeddings_2D[:,1], s=2, cmap="tab10", c=predictions)
        axs[i].set_title(f"Deep Clustering with UMAP-{name} for LSTM", size=8)

    save_path = os.path.join(
            "trained_models",
            "latent_space_plots",
            SEQUENCE_TYPE,
            "f" + str(N_FEATURES),
            f"{autoencoder.__class__.__name__}_h{args.latent_space}_e{args.n_epochs}.png"
        )
    fig.savefig(fname=save_path)
    print(f"saved embedding plot in {save_path}")


