#%% 

import numpy as np
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt

import torch
from umap import UMAP

from aeon.datasets import load_classification
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.LSTM import *

#%%
def load_data(only_digits = True):
    dataset_names = ["ArticularyWordRecognition",
            "AsphaltObstaclesCoordinates",
            "AsphaltPavementTypeCoordinates",
            "AsphaltRegularityCoordinates",
            "AtrialFibrillation",
            "BasicMotions",
            "CharacterTrajectories",
            "Cricket",
            "DuckDuckGeese",
            "EigenWorms",
            "Epilepsy",
            "EthanolConcentration",
            "ERing",
            "FaceDetection",
            "FingerMovements",
            "HandMovementDirection",
            "Handwriting",
            "Heartbeat",
            "InsectWingbeat",
            # "KickVsPunch", # Poorly formatted and very small train size
            "JapaneseVowels",
            "Libras",
            "LSST",
            "MotorImagery",
            "NATOPS",
            "PenDigits",
            "PEMS-SF",
            "PhonemeSpectra",
            "RacketSports",
            "SelfRegulationSCP1",
            "SelfRegulationSCP2",
            "SpokenArabicDigits",
            "StandWalkJump",        
            "UWaveGestureLibrary"]

    if only_digits:
        datasets = load_classification("PenDigits")

    datasets = [load_classification(name) for name in dataset_names]

    return datasets


def plot_reduced_embeddings(embeddings: np.ndarray, labels: int, save_path: str | None = None, model_name:str | None = None):
    """
    Inputs the reduced embeddings and labels, and save a scatter plot
    
    """
    # Map categorical values to integers
    unique_labels, label_ints = np.unique(labels, return_inverse=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=label_ints,
        cmap="tab10",
        s=0.5,
        alpha=0.7
    )
    # Create a legend with the unique label names and their corresponding colors
    handles = []
    for i, label in enumerate(unique_labels):
        handles.append(
            plt.Line2D(
                [], [],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(i % 10),
                markersize=8,
                label=str(label)
            )
        )
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title(f"{model_name if model_name else ''} Embeddings (n={embeddings.shape[0]})")
    plt.legend(handles=handles, title="Label")
    if save_path:
        plt.savefig(save_path, format = "png")
    else:
        plt.show


def print_system_info():
    """Print comprehensive system information for HPC"""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    # CPU cores
    try:
        cpu_cores = os.cpu_count()
    except Exception:
        cpu_cores = "Unknown"
    print(f"CPU cores: {cpu_cores}")

    # GPU info, with PyTorch checks

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs (torch): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"PyTorch GPU {i}: {torch.cuda.get_device_name(i)}")

    # CUDA environment
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    print("=" * 80)

def init_models(
    input_size = 2, # Dimensions/Channels of the input data
    N_EPOCHS = 1,
    LATENT_SPACE = 256,
    validation_split = 0.3,
    VERBOSE = True):
  AutoEncoders = []
  
  AutoEncoders.append(AELSTM(input_size = input_size, 
                             hidden_size=LATENT_SPACE,
                             ))
  
  Trainer = AE_Trainer(max_epochs= N_EPOCHS * 3, verbose = VERBOSE, validation_split=validation_split) #trainer for pytorch models
  return AutoEncoders, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize autoencoder models with specified parameters.")
    parser.add_argument('--n-epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--latent-space', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--dataset', type=str, default="UCR_PenDigits", help='Dataset name')
    parser.add_argument('--validation-split', type=float, default=0.3, help="Fraction of data to be used as validation set")
    # parser.add_argument('--input_size', type=int, default=2, help='Input size (number of channels/dimensions)')
    parser.add_argument('--verbose', action='store_true', help='Verbosity flag')
    return parser.parse_args()

# %%
if __name__ == "__main__":

    args = parse_args()
    print_system_info()
    
    print("="*80)
    print("Running comparative training script with arguments:", args)
    print("="*80)
    
    if args.dataset == "UCR_PenDigits":
        data = load_classification(name="PenDigits")
        data_tensor = UCR_Dataset(data)
        print(f"Loaded PenDigits data, shape {data[0].shape}")
    else:
        raise Exception("No other dataset implemented")

    AutoEncoders, trainer = init_models(input_size=data[0].shape[1] ,
                                        N_EPOCHS=args.n_epochs, 
                                        LATENT_SPACE=args.latent_space,
                                        validation_split= args.validation_split,
                                        VERBOSE=args.verbose)
    reducer = UMAP(n_neighbors=15,
                   n_components= 2)
    
    print("Initialized AutoEncoders:")
    for autoencoder in AutoEncoders:
        print(f"{autoencoder.__class__.__name__}")

    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("trained_models/loss_plots", exist_ok=True)
    os.makedirs("trained_models/embeddings", exist_ok=True)

    train_time_log = {}

    for autoencoder in AutoEncoders:
        print(f"fitting {autoencoder.__class__.__name__}")
        start_time = time.time()
 

        trainer.fit(autoencoder, data_tensor)
        trainer.plot_loss(save_path=f"trained_models/loss_plots/{autoencoder.__class__.__name__}_{args.dataset}_h{args.latent_space}_e{args.n_epochs}.png")
        torch.save(autoencoder.state_dict(), f"trained_models/{autoencoder.__class__.__name__}_{args.dataset}_h{args.latent_space}_e{args.n_epochs}.pth")

        _ , embeddings = autoencoder(data_tensor[:]["data"].to(trainer.device), return_encoding = True)
        embeddings = embeddings.detach().cpu().numpy()
        
        end_time = time.time()
        train_time_log[f"{autoencoder.__class__.__name__}"] = end_time - start_time
        reduced_embeddings = reducer.fit_transform(embeddings)
        plot_reduced_embeddings(
            reduced_embeddings, 
            labels=data[1], save_path=f"trained_models/embeddings/{autoencoder.__class__.__name__}_h{args.latent_space}_e{args.n_epochs}.png", 
            model_name= f"{autoencoder.__class__.__name__}")

    print("="*80)
    print("Run completed")
    for idx, (model, training_time) in enumerate(train_time_log.items()):
        print(f"{model} model trained in {training_time}.\
              Starting loss={AutoEncoders[idx].val_loss_history[0]}.\
                Final Loss={AutoEncoders[idx].val_loss_history[-1]}")
    print("="*80)

