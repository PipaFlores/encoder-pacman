import os

import numpy as np
import pandas as pd
import random
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
    from aeon.clustering.deep_learning import AEResNetClusterer, AEDRNNClusterer, AEDCNNClusterer, BaseDeepClusterer
    from aeon.clustering import DummyClusterer
except ImportError:
    KERAS_AVAILABLE = False

## Reducing
from umap import UMAP
try:
    from pacmap import PaCMAP
except ImportError:
    PACMAP_AVAILABLE = False

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
            reducer: UMAP | PCA | None = None,
            clusterer: HDBSCAN | KMeans | GeomClustering = None,
            similarity_measure: str = "euclidean",
            sequence_type: str = "first_5_seconds",
            validation_method: str | None = "Behavlets",
            features_columns: list[str] = ["Pacman_X", "Pacman_Y"], # Maybe use a different input logic (e.g., geometric_2, geometric_10, Astar, full_state,etc.)
            augmented_visualization: bool = False,
            batch_size: int = 32,
            max_epochs: int = 500,
            latent_dimension: int = 256,
            validation_data_split: int = 0.3,
            using_hpc: bool = False,
            random_seed: int | None = None,
            verbose: bool = False
            ):
        """
        Initializes the PatternAnalysis pipeline with the specified configuration.

        Args:
            reader (PacmanDataReader, optional): Data reader object for loading Pacman data. If None, a new reader is created.
            data_folder (str): Path to the data directory.
            hpc_folder (str): Path to the HPC directory for videos, affinity matrices, and trained models.
            embedder (str | None): Type of embedder to use "LSTM", "DRNN", "DCNN", "ResNet" or None to skip embedding.
            reducer (UMAP | PCA | None): Dimensionality reduction method. If None, defaults to UMAP.
            clusterer (HDBSCAN | KMeans | GeomClustering): Clustering algorithm to use.
            similarity_measure (str): Similarity metric for affinity matrix calculation (euclidean or cosine). Not the one
            used in clustering or reducing, define those in reducer, or clusterer instances.
            sequence_type (str): Type of sequence to analyze (e.g., "first_5_seconds").
            validation (str): Validation method for clusters (e.g., "Behavlets").
            features_columns (list[str]): List of feature column names to use.
            augmented_visualization (bool): Whether to use augmented visualization.
            batch_size (int): Batch size for neural network training.
            max_epochs (int): Maximum number of epochs for training.
            latent_dimension (int): Latent dimension size for embeddings.
            validation_data_split (int): Fraction of data to use for validation.
            using_hpc (bool): Whether to use HPC for parallel computations.
            random_seed (int | None): Random seed for reproducibility.
            verbose (bool): If True, enables verbose logging.

        Sets up the core components of the pipeline, including data reader, embedder, reducer, clusterer,
        and visualization tools. Handles configuration for both geometric and neural network-based clustering.
        """
        
        self.verbose = verbose
        if verbose:
            logger.setLevel("INFO")
        
        # Configuration
        self.sequence_type = sequence_type
        self.validation_method = validation_method ## What kind of method used to assess cluster validity
        self.augmented_visualization = augmented_visualization
        self.hpc_folder = hpc_folder ## for videos and trained models lookups (wherever videos/ affinity_matrices/ and trained_models/ are)
        self.using_hpc = using_hpc ## For parallel computing of affinity matrix
        if random_seed is not None: # For reproducibility
            if TORCH_AVAILABLE:
                torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            # for deep neural networks
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.latent_dimension = latent_dimension
        self.validation_data_split = validation_data_split

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core components
        self.reader = reader if reader is not None else PacmanDataReader(data_folder)
        if isinstance(clusterer, GeomClustering): # Only clusterer.
            
            self.features_columns = ["Pacman_X", "Pacman_Y"] # Always these two features
            logger.info(f"features columns set to Pacman's X,Y coordinates for GeomClustering")
            self.embedder = None
            self.reducer = None
            self.clusterer = clusterer
            self.similarity_measure = clusterer.similarity_measures.measure_type
        else: # Neural-Network embedding -> Reducer -> clusterer
            self.features_columns = features_columns
            self.embedder = self._initialize_deep_embedder(embedder) if embedder is not None else None
            self.reducer = reducer if reducer is not None else UMAP(n_neighbors=15, n_components=2, metric="euclidean", random_state=random_seed)
            self.clusterer = clusterer if clusterer is not None else HDBSCAN(min_cluster_size=20, min_samples=None)
            self.similarity_measure = similarity_measure


        
        
        # Visualization components
        self.gamevisualizer = GameVisualizer(data_folder)
        self.clustervisualizer = ClusterVisualizer()
        
        # Pipeline state
        self.raw_data = None
        self.sequence_data = None
        self.padded_sequence_data = None
        self.trajectory_list = None
        self.gif_path_list = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.affinity_matrix = None
        self.labels = None
        self.validation_encodings = None
        self.validation_labels = None
        self.metadata = None
        self.results = {}

    def _initialize_deep_embedder(self, embedder:str):

        supported = ["LSTM", "DRNN", "DCNN", "ResNet"]
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
        
        if embedder == "DCNN":
            if not KERAS_AVAILABLE:
                raise ModuleNotFoundError(f"using DCNN requires aeon/tensorflow/keras in the environment")
            
            self.using_torch = False
            self.using_keras = True

            return AEDCNNClusterer(
                estimator=DummyClusterer(),
                latent_space_dim=self.latent_dimension,
                n_epochs=self.max_epochs,
                validation_split=self.validation_data_split,
                verbose=self.verbose
            )

    def fit(self, 
            test_dataset: bool = False):
        """
        Main fitting method that runs the complete pipeline.
        """
        logger.info("Starting PatternAnalysis pipeline...")
        
        # Step 1: Load data if not provided
        # TODO maybe remove this step and get the data from reader class.
        # It might greatly improve performance for comparative analysis since each
        # class is loading the data again. Or just leave it to the hpc supremacy.
        
        if test_dataset == False:
            self.raw_data, self.sequence_data, self.padded_sequence_data, self.trajectory_list, self.gif_path_list = self.load_and_slice_data(
                sequence_type=self.sequence_type, make_gif=self.augmented_visualization)
        else:
            from aeon.datasets import load_classification
            logger.info("Loading PenDigits test dataset (10k samples)")
            self.sequence_data, self.validation_labels = load_classification("PenDigits")
            self.sequence_data = self.sequence_data.transpose(0,2,1) # [N, seq_length, features]
            self.padded_sequence_data = self.sequence_data # to accomodate the pipeline

        # Step 2a: load or train embedding model
        
        if self.embedder is not None:
            is_model_trained_, model_path = self._check_model_training_status(return_path=True, test_dataset=test_dataset)
 
            if is_model_trained_:
                logger.info(f"found trained model at {model_path}, loading...")
                self._load_model(model_path)
            else:
                logger.info(f"no trained model found at {model_path}, training...")
                self._train_model(
                    padded_sequence_data=self.padded_sequence_data,
                    test_dataset=test_dataset)
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
            logger.info(f"Performing {self.clusterer.__class__.__name__} clustering")
            self.labels = self.clusterer.fit_predict(self.reduced_embeddings)

        self.labels = self._sort_labels()
        logger.info(f"Clustering complete. Found {len(set(self.labels)) - 1} clusters")

        self.cluster_sizes = None
        self.cluster_centroids = None
        self.trajectory_centroids = None  

        self.clustervisualizer = ClusterVisualizer(
        )

        # Step 4: Validation
        if self.validation_method and self.raw_data is not None:
            self.validation_encodings, self.validation_labels = self.calculate_validation_encodings(self.raw_data)
        # Calculate clustering validation measures (ARI, AMI, NMI) for each validation label set
            self.validation_measures =self.calculate_validation_measures(
                labels = self.labels,
                validation_labels= self.validation_labels
                )
            

        # Step 5: Visualization and Collection of Results
        
        
        # self.summarize()

        # self.plot_results()
        
        logger.info("Pipeline completed successfully!")
        return self

    ### DATA SLICING
    def load_and_slice_data(self, 
                           sequence_type:str = "first_5_seconds",
                           make_gif: bool = False
                           ) -> tuple[pd.DataFrame ,list[np.ndarray], np.ndarray, list[Trajectory]] | tuple[pd.DataFrame ,list[np.ndarray], np.ndarray, list[Trajectory], list[str]]:
        """
        Load and slice data based on the specified sequence type.

        Parameters
        ----------
        sequence_type : str
            Type of sequence to extract. Options are:
                - "first_5_seconds"
                - "whole_level"
                - "last_5_seconds"
                - "first_50_steps" (for debugging)
        make_gif : bool, optional
            Whether to generate GIFs for the sequences.

        Returns
        -------
        raw_data : pd.DataFrame
            Raw data for the selected sequences.
        sequence_data : list[np.ndarray]
            List of game sequences of the chosen type and features.
        padded_sequence_data : np.ndarray
            Fixed-length array of shape [n_samples, max_seq_length, n_features], containing padded sequences for deep learning models.
        trajectory_list : list[Trajectory]
            List of `Trajectory` objects for each sequence, containing metadata for visualization.
        gif_path_list: list[str]
            If `make_gif = True`, returns a list of paths to each sequence's rendered .gif animation, for augmented visualization.
        """
        logger.info(f"Loading and slicing data for {self.sequence_type}...")

        ## TODO add event-based slicing
        
        # Determine slicing parameters based on sequence type
        if sequence_type == "first_5_seconds":
            raw_data, sequence_data, traj_list, gif_paths = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=100, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "whole_level":
            raw_data, sequence_data, traj_list, gif_paths = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=-1, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "last_5_seconds":
            raw_data, sequence_data, traj_list, gif_paths = self.reader.slice_seq_of_each_level(
                start_step=-100, end_step=-1, FEATURES=self.features_columns, make_gif=make_gif
            )
        elif sequence_type == "first_50_steps": # For GeomClustering debugging
            raw_data, sequence_data, traj_list, gif_paths = self.reader.slice_seq_of_each_level(
                start_step=0, end_step=50, FEATURES=self.features_columns, make_gif=make_gif
            )
            
        else:
            raise ValueError(f"Sequence type ({sequence_type}) not valid")
        

        #= Create dataset
        # np.ndarray.shape = [n_samples, max_length, features]. Equal length tensor, padded, for DL models.
        padded_sequence_data = self.reader.padding_sequences(sequence_list=sequence_data)
        
        # list[np.ndarray]
        sequence_data = sequence_data

        # list[Trajectory] // custom class for visualization purposes. Conbtains only Pacman data and level metadata.
        trajectory_list = traj_list
        
        logger.info(f"Data loaded: {len(sequence_data)} sequences with {padded_sequence_data.shape[2]} features")

        return raw_data, sequence_data, padded_sequence_data, trajectory_list, gif_paths
        

    ### EMBEDDING

    def _check_model_training_status(self, return_path=False, test_dataset = False) -> bool | tuple[bool, str]:
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
            "test_data" if test_dataset else "" + f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}" 
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
            self.embedder.load_model(model_path, estimator=None)

        return

    def _train_model(self, 
                     padded_sequence_data: np.ndarray,
                     test_dataset: bool = False):
        """
        Train the embedding model (either PyTorch or Keras-based) on the provided padded sequence data.

        Args:
            padded_sequence_data (np.ndarray): The input data to train the embedding model, of shape [N, seq_length, features].
            test_dataset (bool, optional): If True, indicates that the training is being performed on a test dataset. Defaults to False.

        Returns:
            None. The trained model is saved to disk at the specified model path.
        """

        model_path = os.path.join(
            self.hpc_folder,
            "trained_models",
            self.sequence_type,
            "f" + str(len(self.features_columns)),
            "test_data" if test_dataset else "" + f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}"
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
            data_tensor = PacmanDataset(gamestates = padded_sequence_data)
            trainer.fit(model= self.embedder, data=data_tensor)
            trainer.plot_loss(
                save_path=os.path.join(
                    self.hpc_folder,
                    "trained_models",
                    "loss_plots",
                    self.sequence_type,
                    "f" + str(len(self.features_columns)),
                    "test_data" if test_dataset else "" + f"{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}"
                )
            )
            
        elif self.using_keras:
            self.embedder.save_best_model = True
            self.embedder.best_file_name = model_path + "_best"
            self.embedder.save_last_model = True
            self.embedder.last_file_name = model_path + "_last"

            self.embedder.fit(padded_sequence_data.transpose(0,2,1)) # Transpose to match aeon input format of [n, channels, seq_length]

            self.embedder.plot_loss_keras(
                os.path.join(
                    "trained_models",
                    "loss_plots",
                    self.sequence_type,
                    "f" + str(len(self.features_columns)),
                    "test_data" if test_dataset else "" + "{self.embedder.__class__.__name__}_h{self.latent_dimension}_e{self.max_epochs}.png"
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
        
        else:
            self.reader.level_df 

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
            ## Batch form (Seems unnecessary for aeon modules, they have an internal batch logic)
            # for i in range(0, len(padded_data), self.batch_size):
            #     batch_data = padded_data[i:i+self.batch_size]
            #     batch_embeddings = self.embedder.model_.layers[1].predict(batch_data)
            #     all_embeddings.append(batch_embeddings)
            # embeddings = np.concatenate(all_embeddings, axis=0)

            embeddings = self.embedder.model_.layers[1].predict(padded_data)

            

        
        
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
        
        is_affinity_ok = False
        if os.path.exists(affinity_matrix_path):
            logger.info(f"Using existing affinity matrix ({affinity_matrix_path})")
            self.affinity_matrix = np.loadtxt(affinity_matrix_path, delimiter=',')

            # Verify that the affinity matrix matches the number of trajectories
            if self.affinity_matrix.shape[0] == len(trajectory_list):
                is_affinity_ok = True
            else:
                logger.warning(
                    f"Affinity matrix shape {self.affinity_matrix.shape} does not match number of trajectories {len(trajectory_list)}. Will recalculate."
                )
            
        ## Else, calculate affinity matrix
        if not is_affinity_ok:
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
            logger.info(f"Calculated and saved affinity matrix in {affinity_matrix_path}")
    
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
    
    def calculate_validation_encodings(self, 
                                       raw_data:list[pd.DataFrame]) -> pd.DataFrame:
        """
        Validate clustering results using behavioral patterns.
        
        Args:
            
            
        Returns:
            
        """
        logger.info(f"Validating with {self.validation_method}...")
        
        if self.validation_method == "Behavlets":
            # Use BehavletsEncoding for validation
            behavlets_encoder = BehavletsEncoding(
                data_folder=self.reader.data_folder,
                verbose=self.verbose
            )
            behavlet_types=["Aggression1", ## Hunt close to ghost house
                            # "Aggression3", ## Ghost kills # FIXME
                            "Aggression4", ## Hunt after pill finishes
                            "Caution1", ## Times trapped by ghost
                            "Caution3", ## Close calls
                            ]  # Example behavlets
                        
            # Get behavlet encodings for validation
            validation_encodings = pd.DataFrame()
            # To ensure the order of results aligns with the raw_data list, collect results in a list and concatenate at the end
            validation_rows = []
            for idx, gamestate in enumerate(raw_data):
                try:
                    summary_row = behavlets_encoder.calculate_behavlets_gamestate_slice(
                        gamestates=gamestate, behavlet_type=behavlet_types
                    )
                    validation_rows.append(summary_row)
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate validation for gamestate idx {idx} (level_id {getattr(gamestate[0], 'level_id', 'unknown')})"
                    )
                    validation_rows.append(pd.DataFrame())  # Append empty DataFrame to preserve order
                    continue  # Don't append empty DataFrame

            # Concatenate all rows in the same order as raw_data
            validation_encodings = pd.concat(validation_rows, ignore_index=True)

            # Only keep columns ending with "_value" if they exist
            value_cols = [col for col in validation_encodings.columns if col.endswith("_value")]
            if value_cols:
                validation_encodings = validation_encodings[value_cols]

            # Binarize each column in validation_encodings: >0 -> 1, <=0 -> -1, unless all zeros (then None)
            validation_labels = pd.DataFrame(index=validation_encodings.index)
            for col in validation_encodings.columns:
                col_values = validation_encodings[col].to_numpy()
                if np.sum(col_values) == 0:
                    validation_labels[col] = None
                else:
                    validation_labels[col] = np.where(col_values > 0, 1, -1)


            return validation_encodings, validation_labels
        else:
            raise NotImplementedError(f"Validation method not supported ({self.validation_method})")
        
    
    def calculate_validation_measures(
            self,
            labels : np.ndarray,
            validation_labels : pd.DataFrame
            ): 
        
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
        if validation_labels is not None:
            validation_measures_rows = []
            for col in validation_labels.columns:
                val_labels = validation_labels[col].to_numpy()
                # Only compute if val_labels is not all NaN and not all zeros
                if np.all(pd.isnull(val_labels)) or (np.unique(val_labels).size == 1 and np.unique(val_labels)[0] in [None, np.nan, 0]):
                    ari = np.nan
                    ami = np.nan
                    nmi = np.nan
                else:
                    # Some validation labels may have NaNs, mask those out for fair comparison
                    mask = ~pd.isnull(val_labels)
                    if np.sum(mask) == 0:
                        ari = np.nan
                        ami = np.nan
                        nmi = np.nan
                    else:
                        ari = adjusted_rand_score(labels[mask], val_labels[mask])
                        ami = adjusted_mutual_info_score(labels[mask], val_labels[mask])
                        nmi = normalized_mutual_info_score(labels[mask], val_labels[mask])
                validation_measures_rows.append({
                    "validation_set": col,
                    "ARI": ari,
                    "AMI": ami,
                    "NMI": nmi
                })
            
            validation_measures = pd.DataFrame(validation_measures_rows).set_index("validation_set")

            return validation_measures
        

    def summarize(self):
        """
        Generate summary statistics and results.
        """
        ("Generating summary...")

        # Print out configuration of core modules in the pipeline
        print("\n--- MODULES CONFIGURATION ---")
        print(f"Data Reader: {type(self.reader).__name__}")
        if hasattr(self, 'embedder') and self.embedder is not None:
            print(f"Embedder: {type(self.embedder).__name__}")
        else:
            print("Embedder: None")
        if hasattr(self, 'reducer') and self.reducer is not None:
            print(f"Reducer: {type(self.reducer).__name__}")
        else:
            print("Reducer: None")
        if hasattr(self, 'clusterer') and self.clusterer is not None:
            print(f"Clusterer: {type(self.clusterer).__name__}")
        else:
            print("Clusterer: None")
        print(f"Similarity Measure: {getattr(self, 'similarity_measure', 'N/A')}")
        print(f"Validation Method: {getattr(self, 'validation_method', 'N/A')}")
        print(f"Random Seed: {getattr(self, 'random_seed', 'N/A')}")
        
        self.results = {
            'n_samples': len(self.sequence_data) if self.sequence_data else "N/A",
            'n_features': len(self.features_columns),
            'sequence_type': self.sequence_type,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else "N/A",
            'reduced_dim': self.reduced_embeddings.shape[1] if self.reduced_embeddings is not None else "N/A",
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


        
    def plot_affinity_matrix_overview(self,
                                      axs: list[plt.Axes] | None = None):
        """
        Plot a comprehensive overview of the affinity matrix.

        Creates a figure with 4 subplots showing different aspects of the affinity matrix:
        a) The affinity matrix heatmap
        b) Histogram of distances in the matrix
        c) Bar chart of non-repetitive distance values
        d) Average column values

        Args:
            axs (list[plt.Axes] | None, optional): Array of 4 axes objects to plot on.
                If None, a new figure and axes are created. Defaults to None.
        """
        if axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            show_plot = True
        else:
            fig = None
            show_plot = False
        
        if not isinstance(self.clusterer, GeomClustering):
    
            self.affinity_matrix = self._calculate_affinity_matrix()
            suptitle = f"Embeddings Affinity Matrix Overview - {self.similarity_measure} distance in {self.embedder.__class__.__name__}_{self.reducer.__class__.__name__} latent space"
        else:
            suptitle = f"Trajectory Affinity Matrix Overview - {self.similarity_measure} trajectory similarity measure"

        self.clustervisualizer.plot_affinity_matrix(self.affinity_matrix,
                                              measure_type=self.similarity_measure,
                                              ax=axs[0])
        
        axs[0].set_title("a) " + axs[0].get_title())
        self.clustervisualizer.plot_distance_matrix_histogram(self.affinity_matrix,
                                                        ax=axs[1])
        axs[1].set_title("b) " + axs[1].get_title())
        self.clustervisualizer.plot_non_repetitive_distances_values_barchart(self.affinity_matrix,
                                                                       ax=axs[2])
        axs[2].set_title("c) " + axs[2].get_title())
        self.clustervisualizer.plot_average_column_value(self.affinity_matrix,
                                                   ax=axs[3])
        axs[3].set_title("d) " + axs[3].get_title())

        # Add title to the figure
        if fig is not None:
            fig.suptitle(
                suptitle
            )
            fig.tight_layout()

        if show_plot:
            plt.show()

    def plot_interactive_overview(self, 
                                  plot_only_latent_space: bool = False,
                                  metadata = None , # Not implemented
                                  save_path: str = None,
                                  plot_notebook: bool = False):
        """
        Create an interactive visualization of the affinity matrix and trajectory embeddings.

        This method uses Bokeh to generate an interactive plot that displays:
            - The affinity matrix as a heatmap, visualizing pairwise distances or similarities between trajectories.
            - A 2D embedding visualization of the trajectories (or their centroids), where each point represents a trajectory in reduced latent space.

        The two plots are arranged side by side for comparative analysis.

        Args:
            plot_only_latent_space (bool, optional): If True, only the latent space embedding plot is shown. Defaults to False.
            metadata (optional): Additional metadata to be used for augmented visualization. Not implemented.
            save_path (str, optional): If provided, saves the interactive plot to the specified file path.
            plot_notebook (bool, optional): If True, displays the plot inline in a Jupyter notebook. Defaults to False.

        Returns:
            None. Displays the interactive Bokeh plots either inline (in a notebook) or in a browser window, and optionally saves to file.
        """
        from bokeh.plotting import show, row, output_notebook, output_file
        

        if save_path is not None:
            # If save_path is a filename without a directory, os.path.dirname(save_path) returns ''
            # In that case, don't try to create a directory
            if not save_path.lower().endswith('.html'):
                save_path += '.html'
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            output_file(save_path)

        if isinstance(self.clusterer, GeomClustering):
            # Plot trajectory centroids, as there are no embeddings.
            p1 = self.clustervisualizer.plot_affinity_matrix_bokeh(
                affinity_matrix=self.affinity_matrix, 
                measure_type=self.similarity_measure)

            self.trajectory_centroids = self._calculate_trajectory_centroids()
            
            if self.augmented_visualization:
                p2 = self.clustervisualizer.plot_augmented_trajectories_embedding_bokeh(
                    traj_embeddings=self.trajectory_centroids,
                    gif_path_list=self.gif_path_list,
                    labels=self.labels,
                    metadata=metadata
                )
            else:
                p2 = self.clustervisualizer.plot_trajectories_embedding_bokeh(
                        traj_embeddings=self.trajectory_centroids,
                        labels=self.labels
                    )
            
        else:
            # First calculate affinity_matrix and plot reduced embeddings for dnn based analysis.
            self.affinity_matrix = self._calculate_affinity_matrix()

            p1 = self.clustervisualizer.plot_affinity_matrix_bokeh(
                affinity_matrix=self.affinity_matrix, 
                measure_type=self.similarity_measure)

            if self.augmented_visualization:
                logger.info("plotting augmented")
                p2 = self.clustervisualizer.plot_augmented_trajectories_embedding_bokeh(
                    traj_embeddings=self.reduced_embeddings,
                    gif_path_list=self.gif_path_list,
                    labels=self.labels,
                    metadata=metadata
                )
            else:
                p2 = self.clustervisualizer.plot_trajectories_embedding_bokeh(
                        traj_embeddings=self.reduced_embeddings,
                        labels=self.labels
                    )
            
            p2.xaxis.axis_label = "Reduced Dim. 1"
            p2.yaxis.axis_label = "Reduced Dim. 2"

        if plot_notebook:
            output_notebook()
            
        if plot_only_latent_space:
            show(p2)
        else:
            show(row(p1, p2))

        return
    
    def plot_latent_space_overview(self, 
                                   plot_cluster_centroids = False,
                                   validation_set: str | list | None = None,
                                   all_labels_in_legend: bool = False,
                                   save_path:str = None):
        """
        Plot the trajectory embeddings (or geometrical centroids) colored by their cluster assignments.
        
        This method visualizes the spatial distribution of trajectory centroids,
        with each point colored according to its cluster membership.

        If using embeddings, the centroids plot will only use the first two dimensions.
        """

        if not plot_cluster_centroids:

            if isinstance(self.clusterer, GeomClustering):
                self.trajectory_centroids = self._calculate_trajectory_centroids()  # Geometrical embedding (centroid)
                traj_embeddings = self.trajectory_centroids
                frame_to_maze = True
                xlabel = "X coordinate"
                ylabel = "Y coordinate"
            else:
                traj_embeddings = self.reduced_embeddings
                frame_to_maze = False
                xlabel = "Reduced Latent Dimension 1"
                ylabel = "Reduced Latent Dimension 2"

                
            # Check if validation_set is None or a string (single validation set) One plot
            if validation_set is None or isinstance(validation_set, str):
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                if validation_set is not None:
                    # Get labels for the specified validation set
                    try:
                        labels = self.validation_labels[validation_set].to_numpy()
                    except KeyError:
                        raise KeyError(f"Could not find '{validation_set}'. "
                            f"Possible validation sets are: {list(self.validation_labels.columns)}")
                    # If all values are None, set labels to None
                    if np.all(pd.isnull(labels)):
                        labels = None
                    elif np.sum(labels) == 0:
                        labels = None

                else:
                    labels = self.labels

                self.clustervisualizer.plot_trajectories_embedding(
                    traj_embeddings=traj_embeddings,
                    labels=labels,
                    ax=ax,
                    all_labels_in_legend=all_labels_in_legend,
                    frame_to_maze=frame_to_maze
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            elif isinstance(validation_set, (list, tuple)):
                n_plots = len(validation_set)
                fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 6, 6))
                # Handle multiple validation sets - create subplots for each
                for i, val_set in enumerate(validation_set):
                    # Get labels for this validation set
                    try:
                        val_labels = self.validation_labels[val_set].to_numpy()
                    except KeyError:
                        raise KeyError(f"Could not find '{val_set}'. "
                            f"Possible validation sets are: {list(self.validation_labels.columns)}")
                    
                    # If all values are None, set labels to None
                    if np.all(pd.isnull(val_labels)):
                        val_labels = None
                    elif np.sum(val_labels) == 0:
                        val_labels = None
                    
                    # Use the appropriate axis from the subplot array
                    if isinstance(axs, (list, tuple, np.ndarray)):
                        current_ax = axs[i] if len(axs) > i else axs[-1]
                    else:
                        current_ax = axs
                    
                    # Plot on the current axis
                    self.clustervisualizer.plot_trajectories_embedding(
                        traj_embeddings=traj_embeddings,
                        labels=val_labels,
                        ax=current_ax,
                        all_labels_in_legend=all_labels_in_legend,
                        frame_to_maze=frame_to_maze
                    )
                    
                    # Set labels and title for this subplot
                    current_ax.set_xlabel(xlabel)
                    current_ax.set_ylabel(ylabel)
                    current_ax.set_title(f"Validation: {val_set}")

            
            fig.suptitle(
                f"{self.sequence_type}_"
                f"f{len(self.features_columns)}"
                f"{self.embedder.__class__.__name__}_"
                f"{self.reducer.__class__.__name__}_"
                f"{self.clusterer.__class__.__name__}"
            )
            fig.tight_layout()

            
            plt.show()
        
        elif plot_cluster_centroids: # Consider removing this, is almost useless and code is disgusting                        
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
        
            if isinstance(self.clusterer, GeomClustering):
                self.trajectory_centroids = self._calculate_trajectory_centroids()
                self.cluster_centroids, self.cluster_sizes = self._calculate_cluster_centroids()
                self.clustervisualizer.plot_trajectories_embedding(
                    traj_embeddings=self.trajectory_centroids,
                    labels=labels,
                    ax=axs[0],
                    all_labels_in_legend=all_labels_in_legend,
                    frame_to_maze=True
                )
                axs[0].set_title("a) " + axs[0].get_title())
                axs[0].set_xlabel("X Coordinate")
                axs[0].set_ylabel("Y Coordinate")

                self.clustervisualizer.plot_clusters_centroids(
                    cluster_centroids=self.cluster_centroids,
                    cluster_sizes=self.cluster_sizes,
                    labels=labels,
                    ax=axs[1],
                    frame_to_maze=True
                )
                axs[1].set_title("b) " + axs[0].get_title())
                axs[1].set_xlabel("X Coordinate")
                axs[1].set_ylabel("Y Coordinate")
            else:
                self.cluster_centroids, self.cluster_sizes = self._calculate_cluster_centroids()
                self.clustervisualizer.plot_trajectories_embedding(
                    traj_embeddings=self.reduced_embeddings,
                    labels=labels,
                    ax=axs[0],
                    all_labels_in_legend=all_labels_in_legend,
                    frame_to_maze=False
                )
                axs[0].set_title("a) " + axs[0].get_title())
                axs[0].set_xlabel("Reduced Latent dimension 1")
                axs[0].set_ylabel("Reduced Latent dimension 2")

                self.clustervisualizer.plot_clusters_centroids(
                    cluster_centroids=self.cluster_centroids,
                    cluster_sizes=self.cluster_sizes,
                    labels=labels,
                    ax=axs[1],
                    frame_to_maze=False
                )
                axs[1].set_title("b) " + axs[0].get_title())
                axs[1].set_xlabel("Reduced Latent dimension 1")
                axs[1].set_ylabel("Reduced Latent dimension 2")


            fig.suptitle(f"Latent Space Overview")
            fig.tight_layout()

            
            plt.show()

    def plot_cluster_overview(self, 
                              cluster_id: int, 
                              figsize: tuple[int, int] = (18, 6),
                              darkmode: bool = True,
                              save_path: str = None):

        """
        Plot an overview of a specific cluster showing velocity grid, heatmap and sample trajectories.

        This method creates a figure with 6 subplots arranged in a 2x4 grid:
        - Left column (2 rows): Velocity grid and heatmap of the entire cluster
        - Right column (4 rows): 4 randomly selected sample trajectories from the cluster

        Args:
            cluster_id (int): ID of the cluster to visualize
            figsize (tuple[int, int], optional): Figure size as (width, height). Defaults to (18, 6).
            seed (int): seed for randomly selecting cluster trajectories
        """
        cluster_trajectories = [
            traj for traj, l in zip(self.trajectory_list, self.labels) if l == cluster_id
        ]
        # Create a local random number generator with a fixed seed for reproducibility
        rng = random.Random(42)
        cluster_size = len(cluster_trajectories)
        subset = rng.sample(cluster_trajectories, min(4, cluster_size))

        from matplotlib.gridspec import GridSpec

        self.gamevisualizer = GameVisualizer(darkmode=darkmode)
        fig = plt.figure(figsize=figsize)
        G = GridSpec(2, 4, width_ratios=[2, 2, 1, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(G[:, 0])
        ax2 = fig.add_subplot(G[:, 1])
        ax3 = fig.add_subplot(G[0, 2])
        ax4 = fig.add_subplot(G[0, 3])
        ax5 = fig.add_subplot(G[1, 2])
        ax6 = fig.add_subplot(G[1, 3])

        self.gamevisualizer.plot_velocity_grid(
            trajectory=cluster_trajectories,
            normalize=True,
            ax=ax1,
            title_id=f"Cluster {cluster_id} (n = {cluster_size})",
        )
        self.gamevisualizer.plot_heatmap(
            trajectory=cluster_trajectories,
            normalize=True,
            ax=ax2,
            title_id=f"Cluster {cluster_id}",
        )
        self.gamevisualizer.plot_multiple_trajectories(
            trajectories=subset,
            plot_type="line",
            axs=[ax3, ax4, ax5, ax6],
            show_maze=False,
            metadata_label="level_id",
        )

        return

    def _calculate_affinity_matrix(self):
        """
        Calculate affinity matrix of DNN-based embeddings,
        as such, it uses cosine similarity to calculate pair-wise distances.
        The similarity is calculated on the reduced latent space used for clustering
        (usually with 2 dimensions).
        The affinity matrix is a symmetric matrix where each element (i,j) represents
        the cosine similarity between embedding i and embedding j

        While similar, is not the one used in `GeomClustering`, which uses
        trajectory similarity measures (euclidean, DTW, etc.).

        Returns:
            np.ndarray: Affinity matrix of shape (n, n)
        """
        logger.info("Calculating affinity matrix")
        time_start = time.time()
        num_elements = len(self.reduced_embeddings)
        affinity_matrix = np.zeros((num_elements, num_elements))

        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                v1 = self.reduced_embeddings[i]
                v2 = self.reduced_embeddings[j]
                if self.similarity_measure == "cosine":
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(v1, v2) / (norm1 * norm2)
                    affinity_matrix[i, j] = similarity
                elif self.similarity_measure == "euclidean":
                    distance = np.linalg.norm(v1 - v2)
                    # similarity = 1 / (1 + distance) 
                    ## Compressed between [0,1]. But a bit useless for mere visualization purposes
                    ## Instead is better to keep the linear relationships and use the negative to have
                    ## higher value -> higher similarity. As long as this is ONLY for visualization.
                    ## The clustering and reducers use their own internal metrics, defined for each module
                    ## at initialization.
                    similarity = -distance
                    affinity_matrix[i, j] = similarity

                affinity_matrix[j, i] = affinity_matrix[i, j]

        logger.info(
            f"Affinity matrix calculation complete in {round(time.time() - time_start, 2)} seconds"
        )
        return affinity_matrix


    def _calculate_cluster_centroids(self) -> np.ndarray:
        """
        Calculate the centroids of all clusters. If using GeomClustering, it calculates them
        from trajectories (reducing each to their geom. center), if not, from the reduced embeddings.

        Returns:
            np.ndarray: Array of centroids for each cluster, shape (n_clusters, 2)
            np.ndarray: Array of sizes for each cluster, shape (n_clusters,)
        """
        logger.debug("Calculating cluster centroids")
        cluster_centroids = []
        cluster_sizes = []
        for label in np.unique(self.labels):
            if label != -1:  # Skip noise points
                # Get all trajectories in this cluster
                if isinstance(self.clusterer, GeomClustering):
                    cluster_elements = [
                        traj
                        for traj, l in zip(self.trajectory_list, self.labels)
                        if l == label
                    ]
                    # Calculate mean for each trajectory first [seq_length, 2] -> [,2]
                    cluster_elements = np.array([np.mean(traj,axis=0) for traj in cluster_elements])

                else:
                    cluster_elements = np.array([
                        embedding
                        for embedding, l in zip(self.reduced_embeddings, self.labels)
                        if l == label
                    ])

                cluster_sizes.append(len(cluster_elements))
                
                # Then calculate the mean of all trajectory means
                cluster_centroid = np.mean(cluster_elements, axis=0)
                cluster_centroids.append(cluster_centroid)

        logger.debug("Cluster centroids calculated")
        return np.array(cluster_centroids), np.array(cluster_sizes)

    def _calculate_trajectory_centroids(self) -> list[np.ndarray]:
        """
        Calculate the geometrical centroids of all trajectories for dimensionality reduction and plotting.

        Returns:
            List[np.ndarray]: List of centroids for each trajectory
        """
        logger.debug("Calculating trajectory centroids")
        centroids = [np.mean(trajectory, axis=0) for trajectory in self.trajectory_list]
        centroids_array = np.array(centroids)
        if centroids_array.ndim == 1:
            centroids_array = centroids_array.reshape(
                -1, 2
            )  # Reshape to (n_trajectories, 2)
        logger.debug("Trajectory centroids calculated")
        return centroids_array
    
    # def get_cluster_samples(self, cluster_id: int, n_samples: int = 5) -> list[int]:
    #     """
    #     Get sample indices from a specific cluster.
        
    #     Args:
    #         cluster_id: ID of the cluster
    #         n_samples: Number of samples to return
            
    #     Returns:
    #         List of sample indices
    #     """
    #     if self.labels is None:
    #         return []
        
    #     cluster_indices = np.where(self.labels == cluster_id)[0]
    #     n_samples = min(n_samples, len(cluster_indices))
        
    #     return np.random.choice(cluster_indices, n_samples, replace=False).tolist()

    # def export_results(self, export_path: str):
    #     """
    #     Export analysis results to file.
        
    #     Args:
    #         export_path: Path to save results
    #     """
    #     import pickle
        
    #     export_data = {
    #         'results': self.results,
    #         'labels': self.labels,
    #         'embeddings': self.embeddings,
    #         'reduced_embeddings': self.reduced_embeddings,
    #         'metadata': self.metadata,
    #         'configuration': {
    #             'sequence_type': self.sequence_type,
    #             'features_columns': self.features_columns,
    #             'validation_method': self.validation_method,
    #             'deep_embedding': self.deep_embedding
    #         }
    #     }
        
    #     with open(f"{export_path}.pkl", 'wb') as f:
    #         pickle.dump(export_data, f)
        
    #     print(f"Results exported to {export_path}.pkl")

