
import os
import sys
import numpy as np
import pandas as pd
import torch
import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import argparse
from umap import UMAP
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import AE_Trainer, AELSTM
from src.datahandlers import PacmanDataset, PacmanDataReader, Trajectory

def parse_args():
    parser = argparse.ArgumentParser(description="Initialize autoencoder models with specified parameters.")
    parser.add_argument('--n-epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--latent-space', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--validation-split', type=float, default=0.3, help="Fraction of data to be used as validation set")
    parser.add_argument('--features', type=str, default= "Pacman", help="Which combination of features to use")
    parser.add_argument('--context', type=int, default=0, help= "number of context frames to be added before and after an event interval (such as pacman attack mode)")
    # parser.add_argument('--input_size', type=int, default=2, help='Input size (number of channels/dimensions)')
    parser.add_argument('--verbose', action='store_true', help='Verbosity flag')
    parser.add_argument('--sequence-type', type= str, default="", help= "On what type of sequences to train the model, see source code")
    parser.add_argument('--logging-comment', type= str, default="", help= "Any special comment to be sent to wandb (if logging)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Training LSTM with")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

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
    elif args.features == "Ghost_Distances":
        FEATURES=[
        'Ghost1_distance',
        'Ghost2_distance', 
        'Ghost3_distance', 
        'Ghost4_distance'
        ]
    else:
        raise ValueError(f"Unknown features selection: {args.features}")

    N_FEATURES = len(FEATURES)

    SEQUENCE_TYPE = args.sequence_type

    if SEQUENCE_TYPE == "": ## hard-coded type for non-bash based iterations
        SEQUENCE_TYPE = "first_5_seconds"

    if SEQUENCE_TYPE == "first_5_seconds":
        raw_sequence_list, _ = reader.slice_seq_of_each_level(
            start_step=0, end_step=100, make_gif=False
        )
    elif SEQUENCE_TYPE == "whole_level":
        raw_sequence_list, _ = reader.slice_seq_of_each_level(
            start_step=0, end_step=-1, make_gif=False
        )
    elif SEQUENCE_TYPE == "last_5_seconds":
        raw_sequence_list, _ = reader.slice_seq_of_each_level(
            start_step=-100, end_step=-1, make_gif=False
        )
    elif SEQUENCE_TYPE == "pacman_attack":
        raw_sequence_list, _ = reader.slice_attack_modes(
            CONTEXT=args.context
        )
    else:
        raise ValueError(f"Sequence type ({SEQUENCE_TYPE}) not valid")

    filtered_sequence_list = [sequence[FEATURES].to_numpy() for sequence in raw_sequence_list]
    X_padded = reader.padding_sequences(sequence_list=filtered_sequence_list)

    ghost_distance_indices = []
    for i, col in enumerate(FEATURES):
        if col.startswith('Ghost') and col.endswith('_distance'):
            ghost_distance_indices.append(i)
    
    if ghost_distance_indices:
        # Sort only the ghost distance columns
        for i in range(X_padded.shape[0]):  # For each sequence
            for j in range(X_padded.shape[1]):  # For each time step
                # Sort only the ghost distance values at this time step
                ghost_values = X_padded[i, j, ghost_distance_indices]
                sorted_ghost_values = np.sort(ghost_values)
                X_padded[i, j, ghost_distance_indices] = sorted_ghost_values


    data_tensor = PacmanDataset(X_padded)
    data_tensor[:]["data"].to(device)

    print(f"loaded data tensor of shape {data_tensor.gamestates.shape}")

    ### TRAIN MODEL
    model_path = os.path.join(
        "trained_models",
        SEQUENCE_TYPE,
        "f" + str(N_FEATURES)
    )

    autoencoder = AELSTM(input_size=data_tensor[0]["data"].shape[1], hidden_size=args.latent_space)

    if WANDB_AVAILABLE:

        run = wandb.init(
                project = "pacman",
                config= {
                    "sequence_type": args.sequence_type,
                    "n_features": len(FEATURES),
                    "features_columns": FEATURES,
                    "feature_set":args.features,
                    
                    # Training hyperparameters
                    "max_epochs": args.n_epochs,
                    "batch_size": 32,
                    "latent_dimension": args.latent_space,
                    "validation_data_split": args.validation_split,
                    # Model architecture
                    "embedder_type": autoencoder.__class__.__name__,
                    "comment": args.logging_comment
                },
                name=f"{args.sequence_type}_{args.features}_{datetime.datetime.now().strftime('%m_%d_%H_%M')}",
                tags=[args.sequence_type, autoencoder.__class__.__name__]
            )
    else:
        run = None

    trainer = AE_Trainer(max_epochs=args.n_epochs, 
                        batch_size=32, 
                        validation_split=args.validation_split,
                        save_model=True,
                        best_path=os.path.join(model_path, f"AELSTM_h{args.latent_space}_e{args.n_epochs}_best.pth"),
                        last_path=os.path.join(model_path, f"AELSTM_h{args.latent_space}_e{args.n_epochs}_last.pth"),
                        wandb_run=run)


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
    reducer = UMAP(n_neighbors=15, n_components=2, metric="euclidean") # default parameters, just explicit 
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
    fig, axs = plt.subplots(1, len(labels.values()), figsize=(10 * len(labels.values()), 10))

    # Ensure axs is always a list
    if len(labels.values()) == 1:
        axs = [axs]

    for i, (name, predictions) in enumerate(labels.items()):
        axs[i].scatter(embeddings_2D[:,0], embeddings_2D[:,1], s=2, cmap="tab20", c=predictions)
        axs[i].set_title(f"Deep Clustering with UMAP-{name} for LSTM", size=8)




    save_path = os.path.join(
            "trained_models",
            "latent_space_plots",
            SEQUENCE_TYPE,
            "f" + str(N_FEATURES),
            f"{autoencoder.__class__.__name__}_h{args.latent_space}_e{args.n_epochs}.png"
        )

    fig.savefig(fname=save_path)
    if WANDB_AVAILABLE:
        run.log({"latent_space_plot": fig})
        run.finish()
        # data = [[x, y, c] for (x, y, c) in zip(embeddings_2D[:,0], embeddings_2D[:,1], predictions)]

        # table = wandb.Table(data=data, columns = ["Dim1", "Dim2", "labels"])

        # run.log({"chart" : wandb.plot.scatter(table, "Dim1", "Dim2",
        #                      title="Latent space plot")
        # })

    print(f"saved embedding plot in {save_path}")


