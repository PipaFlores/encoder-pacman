
import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse

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
    parser.add_argument('--n-epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--latent-space', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--validation-split', type=float, default=0.3, help="Fraction of data to be used as validation set")
    # parser.add_argument('--input_size', type=int, default=2, help='Input size (number of channels/dimensions)')
    parser.add_argument('--verbose', action='store_true', help='Verbosity flag')
    parser.add_argument('--sequence-type', type= str, default="", help= "On what type of sequences to train the model, see source code")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    reader = PacmanDataReader(data_folder="../data", verbose=args.verbose)
    reader.gamestate_df.columns

    SEQUENCE_TYPE = args.sequence_type

    if SEQUENCE_TYPE == "": ## hard-coded type for non-bash based iterations
        SEQUENCE_TYPE = "first_5_seconds"

    if SEQUENCE_TYPE == "first_5_seconds":
        sequence_list, meta, traj_list = slice_seq_of_each_level(reader, start_step=0, end_step=100)
    elif SEQUENCE_TYPE == "whole_level":
        sequence_list, meta, traj_list = slice_seq_of_each_level(reader, start_step=0, end_step=-1)
    elif SEQUENCE_TYPE == "last_5_seconds":
        sequence_list, meta, traj_list = slice_seq_of_each_level(reader, start_step=-100, end_step=-1)
    else:
        raise ValueError(f"Sequence type ({SEQUENCE_TYPE}) not valid")

    X_padded = padding_sequences(sequence_list=sequence_list)

    data_tensor = PacmanDataset(X_padded)
    data_tensor.gamestates.shape

    best_save_path = f"trained_models/pacman_aelstm_{SEQUENCE_TYPE}_h{args.latent_space}_e{args.n_epochs}_best.pth"
    last_save_path = f"trained_models/pacman_aelstm_{SEQUENCE_TYPE}_h{args.latent_space}_e{args.n_epochs}_last.pth"
    trainer = AE_Trainer(max_epochs=args.n_epochs, 
                        batch_size=32, 
                        validation_split=args.validation_split,
                        save_model=True,
                        best_path=best_save_path,
                        last_path=last_save_path)

    autoencoder = AELSTM(input_size=data_tensor[0]["data"].shape[1], hidden_size=256)


    os.makedirs("trained_models", exist_ok=True)
    trainer.fit(autoencoder, data_tensor)
    print(f"Best Model saved to {best_save_path}")

    trainer.plot_loss(f"trained_models/loss_plots/pacman_aelstm_{SEQUENCE_TYPE}_h{args.latent_space}_e{args.n_epochs}.png")
    print(f"trained_models/loss_plots/pacman_aelstm_{SEQUENCE_TYPE}_h{args.latent_space}_e{args.n_epochs}.png")