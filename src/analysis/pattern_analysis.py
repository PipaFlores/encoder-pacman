import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from src.utils import setup_logger

### Data Handling
from src.datahandlers import PacmanDataReader, PacmanDataset, Trajectory

### Embedding 
### trying lazy import first to avoid hpc environment issues. 
## If torch, no keras/tensorflow imports are required, and the opposite.
try:
    import torch
    TORCH_AVAILABLE = True
    from src.models import LSTM, AE_Trainer, AELSTM
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow # check backend
    KERAS_AVAILABLE = True
    from aeon.clustering.deep_learning import AEResNetClusterer, AEDRNNClusterer, BaseDeepClusterer
    from aeon.clustering import DummyClusterer
except ImportError:
    KERAS_AVAILABLE = False

## Reducing
from umap import UMAP
from pacmap import PaCMAP
from sklearn.decomposition import PCA

## Clustering
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from src.analysis import GeomClustering

# Validation
from src.analysis import BehavletsEncoding

## Visualization
from src.visualization import GameVisualizer, ClusterVisualizer

logger = setup_logger(__name__)
class PatternAnalysis:
    """
    A class that handles the whole Pattern extraction and analysis pipeline in a modular style.
    It encompasses several steps, from embedding to visualization and statistics.
    Each step is controlled by its own class.

    Pipeline:
    1. Data loading and slicing 
    2. Embedding generation (deep learning or geometric)
    3. Dimensionality reduction
    4. Clustering 
    5. Validation (using behavlets or other methods)
    6. Visualization and summarization
    """

    def __init__(
            self, 
            reader: PacmanDataReader = None,
            data_folder: str = "../data",
            hpc_folder: str = "../hpc",
            embedder: str | None = "LSTM",
            reducer: UMAP | PaCMAP | PCA | None = None,
            clusterer: HDBSCAN | KMeans | GeomClustering = None,
            sequence_type: str = "first_5_seconds",
            validation: str = "Behavlets",
            features_columns: list[str] = ["Pacman_X", "Pacman_Y"],
            augmented_visualization: bool = False,
            batch_size: int = 32,
            max_epochs: int = 500,
            latent_dimension: int = 256,
            validation_data_split: int = 0.3,
            using_hpc: bool = False,
            verbose: bool = False
            ):
        
        self.verbose = verbose
        if verbose:
            logger.setLevel("INFO")
        
        # Configuration
        self.sequence_type = sequence_type
        self.validation_method = validation ## What kind of method used to assess cluster validity
        self.augmented_visualization = augmented_visualization
        self.hpc_folder = hpc_folder ## for videos and trained models lookups (wherever videos/ affinity_matrices/ and trained_models/ are)
        self.using_hpc = using_hpc ## For parallel computing of affinity matrix

        # Default features if none provided
        self.features_columns = features_columns if features_columns is not None else [
            "Pacman_X", "Pacman_Y"
        ]
            # for deep neural networks
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.latent_dimension = latent_dimension
        self.validation_data_split = validation_data_split

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core components
        self.reader = reader if reader is not None else PacmanDataReader(data_folder)
        self.embedder = self._initialize_deep_embedder(embedder) if embedder is not None else None
        self.reducer = reducer if reducer is not None else UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        self.clusterer = clusterer if clusterer is not None else HDBSCAN(min_cluster_size=20, min_samples=None)
        
        
        # Visualization components
        self.gamevisualizer = GameVisualizer(data_folder)
        self.clustervisualizer = None  # Will be initialized after clustering
        
        # Pipeline state
        self.raw_data = None
        self.sequence_data = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.validation_labels = None
        self.metadata = None
        self.results = {}

    def _initialize_deep_embedder(self, embedder:str):

        supported = ["LSTM", "DRNN","ResNet"]
        if embedder not in supported:
            raise ValueError(f"Embedder {embedder} is not one of the supported embedding deep networks ({supported})")
        
        if embedder == "LSTM":
            if not TORCH_AVAILABLE:
                raise ModuleNotFoundError(f"Using LSTM requires torch in the environment")

            self.using_torch = True
            self.using_keras = False
            return AELSTM(
                input_size= len(self.features_columns),
                hidden_size=256,
            ) # other params defined in trainer during fitting stage.
        
        if embedder == "DRNN":
            if not KERAS_AVAILABLE:
                raise ModuleNotFoundError(f"using DRNN requires aeon/tensorflow/keras in the environment")
            
            self.using_torch = False
            self.using_keras = True

            return AEDRNNClusterer(
                estimator=DummyClusterer(),
                latent_space_dim=self.latent_dimension,
                n_epochs=self.max_epochs,
                validation_split=self.validation_data_split,
                verbose=self.verbose
            )
        if embedder == "ResNet":
            if not KERAS_AVAILABLE:
                raise ModuleNotFoundError(f"using ResNet requires aeon/tensorflow/keras in the environment")
            
            self.using_torch = False
            self.using_keras = True

            self.latent_dimension = 128 # ResNet latent_space fixed to 128

            return AEResNetClusterer(
                estimator=DummyClusterer(),
                n_epochs=self.max_epochs,
                validation_split=self.validation_data_split,
                verbose=self.verbose                
            )

    def fit(self):
        """
        Main fitting method that runs the complete pipeline.
        """
        logger.info("Starting PatternAnalysis pipeline...")
        
        # Step 1: Load data if not provided
        
        self.raw_data, self.sequence_data, self.padded_sequence_data, self.trajectory_list = self.load_and_slice_data(
            sequence_type=self.sequence_type, make_gif=self.augmented_visualization)

        # Step 2a: load or train embedding model
        
        if self.embedder is not None:
            is_model_trained_, model_path = self._check_model_training_status(return_path=True)
 
            if is_model_trained_:
                logger.info(f"found trained model at {model_path}, loading...")
                self._load_model(model_path)
            else:
                logger.info(f"no trained model found, training...")
                self._train_model()
                logger.info(f"model trained and saved at {model_path}")

        # Step 2b: Generate embeddings 
        # (If not conducting geometric clustering, which instead uses trajectory similarity measures)
        if not isinstance(self.clusterer, GeomClustering):
            self.embeddings = self.embed(self.padded_sequence_data)
            
            # Step 2c: Reduce dimensions if embeddings are high-dimensional
            if self.embeddings.shape[1] > 2:
                logger.info("Reducing dimensionality...")
                self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
            else:
                self.reduced_embeddings = self.embeddings

        
        # Step 3: Perform clustering
        if isinstance(self.clusterer, GeomClustering):
            logger.info(f"Performing geometrical clustering with similarity measure: {self.clusterer.similarity_measures.measure_type}")
            self.labels = self._geom_clustering(self.trajectory_list)
        else:
            ## TODO: Is it worth to calculate affinity matrices (cosine similarity) for
            ## embeddings-based clustering?
            logger.info(f"Performing {self.clusterer.__class__.__name__} clustering")
            self.labels = self.clusterer.fit_predict(self.reduced_embeddings)

        self.labels = self._sort_labels()
        logger.info(f"Clustering complete. Found {len(set(self.labels)) - 1} clusters")

        ## TODO Calculate centroids, sizes (cluster AND trajectories)?
        self.cluster_sizes = None
        self.cluster_centroids = None
        self.trajectory_centroids = None  

        self.clustervisualizer = ClusterVisualizer(
            affinity_matrix = None, # FIXME, how to do it when no affinity is calculated / make it optional
            labels = self.labels,
            trajectories = self.trajectory_list,
            measure_type = None # FIXME // make optional
        )

        
        # Step 4: Validation
        # if self.validation_method and self.raw_data is not None:
        #     self.validation_labels = self.validate(self.raw_data)
        
        # Step 5: Visualization and Collection of Results
        # self.clustervisualizer = ClusterVisualizer()
        
        # self.summarize()
        
        print("Pipeline completed successfully!")
        return self

    ### DATA SLICING
    def load_and_slice_data(self, 
                           sequence_type:str = "first_5_seconds",
                           make_gif: bool = False) -> PacmanDataset:
        """
        Load and slice data based on the specified sequence type.

        Args:
            sequence_type (str): Type of sequence to extract ("first_5_seconds", "whole_level", or "last_5_seconds").
            make_gif (bool): Whether to generate GIFs for the sequences.

        Outcome:
            Sets the following attributes:
                self.sequence_data: List of game sequences of the chosen type and features.
                self.padded_sequence_data: Fixed-length `np.ndarray` of shape `[n_samples, max_seq_length, n_features]`,
                    containing padded sequences for deep learning models.
                self.trajectory_list: List of `Trajectory` objects for each sequence, containing metadata for
                    `GameVisualization` plotting methods.
        """
        logger.info(f"Loading and slicing data for {self.sequence_type}...")

        ## TODO add event-based slicing
        
        # Determine slicing parameters based on sequence type
        if sequence_type == "first_5_seconds":
            raw_data, sequence_data, traj_list, make_gif = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=100, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "whole_level":
            raw_data, sequence_data, traj_list, make_gif = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=-1, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "last_5_seconds":
            raw_data, sequence_data, traj_list, make_gif = self.reader.slice_seq_of_each_level(
                start_step=-100, end_step=-1, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "first_50_steps": # For GeomClustering debugging
            raw_data, sequence_data, traj_list, make_gif = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=50, FEATURES=self.features_columns, make_gif=make_gif
            )
            
        else:
            raise ValueError(f"Sequence type ({sequence_type}) not valid")
        

        #= Create dataset
        # [n_samples, max_length, features] Equal length tensor, padded, for DL models.
        padded_sequence_data = self.reader.padding_sequences(sequence_list=sequence_data)
        
        # list[np.ndarray]
        sequence_data = sequence_data

        # list[Trajectory] // custom class for visualization purposes. Conbtains only Pacman data and level metadata.
        trajectory_list = traj_list
        
        logger.info(f"Data loaded: {len(sequence_data)} sequences with {padded_sequence_data.shape[2]} features")

        return raw_data, sequence_data, padded_sequence_data, trajectory_list

    ### EMBEDDING

    def _check_model_training_status(self, return_path=False) -> bool | tuple[bool, str]:
        """
        Check if a trained model already exists for the current configuration.
        
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = os.path.join(
            self.hpc_folder,
            "trained_models",
            self.sequence_type,
            "f" + str(len(self.features_columns)),
            f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}"
        )
        
        if self.using_torch:
            model_path += "_best.pth"
        elif self.using_keras:
            model_path += "_best.keras"

        if return_path:
            return os.path.exists(model_path) , model_path

        return os.path.exists(model_path)
    
    def _load_model(self, model_path):
        
        if self.using_torch:
            self.embedder.load_state_dict(torch.load(model_path, map_location=self.device))
            self.embedder.eval()

        elif self.using_keras:
            model_path += ".keras"
            self.embedder.load_model(model_path)

        return

    def _train_model(self):

        model_path = os.path.join(
            self.hpc_folder,
            "trained_models",
            self.sequence_type,
            "f" + str(len(self.features_columns)),
            f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}"
        )
    
        if self.using_torch:
            trainer = AE_Trainer(
                max_epochs= self.max_epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_data_split,
                verbose=self.verbose,
                save_model=True,
                best_path=model_path + "_best.pth",
                last_path=model_path + "_last.pth",
                )
            data_tensor = PacmanDataset(gamestates = self.padded_sequence_data)
            trainer.fit(model= self.embedder, data=data_tensor)
            trainer.plot_loss(
                save_path=os.path.join(
                    self.hpc_folder,
                    "trained_models",
                    "loss_plots",
                    self.sequence_type,
                    "f" + str(len(self.features_columns)),
                    f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}"
                )
            )
            
        elif self.using_keras:
            self.embedder.save_best_model = True
            self.embedder.best_file_name = model_path + "_best"
            self.embedder.save_last_model = True
            self.embedder.last_file_name = model_path + "_last"

            self.embedder.fit(self.padded_sequence_data.transpose(0,2,1)) # Transpose to match aeon input format of [n, channels, seq_length]

            self.embedder.plot_loss_keras(
                os.path.join(
                    "trained_models",
                    "loss_plots",
                    self.sequence_type,
                    "f" + str(len(self.features_columns)),
                    f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}.png"
                )
            )       
            
    def embed(self, padded_data: np.ndarray) -> np.ndarray:
        """
        Generate embeddings from the input data.
        If geometric clustering is used, no embeddings are produced. Instead
        a similarity/affinity matrix is calculated in the clustering step using
        trajectory similarity measures (only 2 Features [Pacman_X, Pacman_Y]).
        
        Args:
            padded_data: Input data of same-length sequences (padded)
            
        Returns:
            Embedding vectors as numpy array [n_samples, embeddings_dimensionality]
        """
        logger.info("Generating embeddings...")
        
        if self.embedder is not None:
            # Deep learning embeddings
            return self._generate_deep_embeddings(padded_data)

        # else:
        #     # Use raw features as embeddings. 
        # TODO: implement something, either flattening sequences or using higher-level feats
        # such as highest-score, avg. distance to ghosts, whatever. Not a main concern, yet
        #     # FIXME the data is never inputted as a Tensor, so it should be constructed here.
        #     logger.info("Using raw features as embeddings")
        #     if isinstance(data, PacmanDataset):
        #         # Flatten sequences or use summary statistics
        #         embeddings = []
        #         for i in range(len(data)):
        #             sequence = data[i]["data"].numpy()
        #             mask = data[i]["mask"].numpy()
                    
        #             # Use mean of valid timesteps as embedding
        #             valid_sequence = sequence[mask.astype(bool)]
        #             if len(valid_sequence) > 0:
        #                 embedding = np.mean(valid_sequence, axis=0)
        #             else:
        #                 embedding = np.zeros(sequence.shape[-1])
        #             embeddings.append(embedding)
                
        #         return np.array(embeddings)
        #     else:
        #         return np.array(data)

    def _generate_deep_embeddings(self, padded_data: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using deep learning models.
        
        Args:
            padded_data: np.ndarray. sequences of equal length, or padded.
            
        Returns:
            Deep embeddings as numpy array [n_samples, embeddings_dimensionality]
        """
        all_embeddings = []
        
        if self.using_torch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_tensor = PacmanDataset(gamestates=padded_data)
            self.embedder.to(device).eval()
            with torch.no_grad():
                for i in range(0, len(data_tensor), self.batch_size):
                    batch_data = data_tensor[i:i+self.batch_size]["data"].to(device)
                    
                    if hasattr(self.embedder, 'encode'):
                        # For AELSTM and similar torch models
                        batch_embeddings = self.embedder.encode(batch_data)
                    else:
                        raise AttributeError("Torch module has no .encode() method to produce embeddings.")
                    
                    all_embeddings.append(batch_embeddings.cpu())
            
            embeddings = torch.cat(all_embeddings, dim=0)
            embeddings = embeddings.detach().numpy()

        if isinstance(self.embedder, BaseDeepClusterer): ## aeon implementations

            for i in range(0, len(padded_data), self.batch_size):
                batch_data = padded_data[i:i+self.batch_size]
                batch_embeddings = self.embedder.model_.layers[1].predict(batch_data)
                all_embeddings.append(batch_embeddings)

            embeddings = np.concatenate(all_embeddings, axis=0)
        
        
        return embeddings

    ### CLUSTERING

    def _geom_clustering(self, trajectory_list) -> np.ndarray:
        """
        Performs geometric-HDBSCAN clustering on trajectory list.
        It follows the main pipeline on `GeomClustering` but does not
        use the `.fit()` method. This avoids the creation of duplicates affinity matrices, cluster centroids 
        and labels.
        
        """
        
        ## If exists, load affinity matrix
        affinity_matrix_path = os.path.join(
                self.hpc_folder,
                "affinity_matrices",
                self.sequence_type,
                f"{self.clusterer.similarity_measures.measure_type}_affinity_matrix.csv"
                )
        if os.path.exists(affinity_matrix_path):
            logger.info(f"Using existing affinity matrix ({affinity_matrix_path})")
            self.affinity_matrix = np.loadtxt(affinity_matrix_path, delimiter=',')
        ## Else, calculate affinity matrix
        else:
            os.makedirs(
                    os.path.join(
                        self.hpc_folder,
                        "affinity_matrices",
                        self.sequence_type
                    ),
                    exist_ok = True
                )
            if self.using_hpc == True:
                self.affinity_matrix = self.clusterer.calculate_affinity_matrix_parallel_cpu(
                    trajectories= trajectory_list, n_jobs=None, chunk_size_multiplier=1
                )
            else:
                self.affinity_matrix = self.clusterer.calculate_affinity_matrix(
                        trajectories = trajectory_list)
                
            np.savetxt(
                    affinity_matrix_path,
                    self.affinity_matrix,
                    delimiter=",",
                )
    
        labels = self.clusterer.clusterer.fit_predict(self.affinity_matrix)

        return labels
            
        
    def _sort_labels(self) -> np.ndarray:
        """
        Sort cluster labels based on the number of trajectories in each cluster and
        remap labels so that the largest cluster has label 0, second largest has label 1, etc.
        Noise points (label -1) remain unchanged.

        Returns:
            np.ndarray: New array of remapped labels
        """
        logger.debug("Sorting and remapping cluster labels by size")

        if self.labels.size == 0:
            logger.warning("No clusters to sort")
            return np.array([])

        unique_labels = np.unique(self.labels)
        cluster_sizes = []

        # Calculate size of each cluster (excluding noise)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_size = np.sum(self.labels == label)
                cluster_sizes.append((label, cluster_size))

        # Sort clusters by size in descending order
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Create mapping from old labels to new labels
        new_labels = np.copy(self.labels)
        for new_label, (old_label, size) in enumerate(cluster_sizes):
            logger.debug(
                f"Remapping cluster {old_label} (size {size}) to label {new_label}"
            )
            new_labels[self.labels == old_label] = new_label

        logger.debug(f"Remapped {len(cluster_sizes)} clusters")
        return new_labels
    
    
    def validate(self, gamestates: list[np.ndarray]) -> dict[str,]:
        """
        Validate clustering results using behavioral patterns.
        
        Args:
            gamestates: Raw gamestate sequences
            
        Returns:
            Validation results dictionary
        """
        return {}
        ## TODO implement
        print(f"Validating with {self.validation_method}...")
        
        if self.validation_method == "Behavlets":
            # Use BehavletsEncoding for validation
            behavlets_encoder = BehavletsEncoding(
                data_folder=self.reader.data_folder,
                behavlet_types=["Aggression1", "Caution1"]  # Example behavlets
            )
            
            # Get behavlet encodings for validation
            validation_results = {}
            for i, gamestate in enumerate(gamestates):
                if hasattr(self, 'metadata') and self.metadata and i < len(self.metadata):
                    level_id = self.metadata[i].get('level_id')
                    if level_id:
                        # This would need to be implemented based on BehavletsEncoding interface
                        # validation_results[level_id] = behavlets_encoder.encode_level(level_id)
                        pass
            
            return validation_results
        else:
            return {}

    def summarize(self):
        """
        Generate summary statistics and results.
        """
        print("Generating summary...")
        
        self.results = {
            'n_samples': len(self.sequence_data) if self.sequence_data else 0,
            'n_features': len(self.features_columns),
            'sequence_type': self.sequence_type,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'reduced_dim': self.reduced_embeddings.shape[1] if self.reduced_embeddings is not None else 0,
        }
        
        if self.labels is not None:
            unique_labels = np.unique(self.labels)
            n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
            n_noise = np.sum(self.labels == -1)
            
            self.results.update({
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_sizes': {
                    int(label): int(np.sum(self.labels == label)) 
                    for label in unique_labels
                }
            })
        
        print("\n=== PATTERN ANALYSIS SUMMARY ===")
        for key, value in self.results.items():
            if key != 'cluster_sizes':
                print(f"{key}: {value}")
        
        if 'cluster_sizes' in self.results:
            print("Cluster sizes:")
            for label, size in self.results['cluster_sizes'].items():
                cluster_name = f"Cluster {label}" if label >= 0 else "Noise"
                print(f"  {cluster_name}: {size} samples")

    def plot_results(self, save_path: str = None, figsize: tuple[int, int] = (10, 8)):
        """
        Plot clustering results.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.reduced_embeddings is None or self.labels is None:
            print("No results to plot. Run fit() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        scatter = plt.scatter(
            self.reduced_embeddings[:, 0], 
            self.reduced_embeddings[:, 1],
            c=self.labels, 
            cmap='tab10', 
            s=30,
            alpha=0.7
        )
        
        plt.colorbar(scatter)
        plt.title(f'Pattern Analysis Results - {self.sequence_type}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Add cluster statistics as text
        if 'n_clusters' in self.results:
            stats_text = f"Clusters: {self.results['n_clusters']}\n"
            stats_text += f"Noise points: {self.results['n_noise_points']}"
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def get_cluster_samples(self, cluster_id: int, n_samples: int = 5) -> list[int]:
        """
        Get sample indices from a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            n_samples: Number of samples to return
            
        Returns:
            List of sample indices
        """
        if self.labels is None:
            return []
        
        cluster_indices = np.where(self.labels == cluster_id)[0]
        n_samples = min(n_samples, len(cluster_indices))
        
        return np.random.choice(cluster_indices, n_samples, replace=False).tolist()

    def export_results(self, export_path: str):
        """
        Export analysis results to file.
        
        Args:
            export_path: Path to save results
        """
        import pickle
        
        export_data = {
            'results': self.results,
            'labels': self.labels,
            'embeddings': self.embeddings,
            'reduced_embeddings': self.reduced_embeddings,
            'metadata': self.metadata,
            'configuration': {
                'sequence_type': self.sequence_type,
                'features_columns': self.features_columns,
                'validation_method': self.validation_method,
                'deep_embedding': self.deep_embedding
            }
        }
        
        with open(f"{export_path}.pkl", 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"Results exported to {export_path}.pkl")

