# Encoder Pacman

An unsupervised learning pipeline for discovering structure in the latent space of Pacman gameplay trajectories, and using that structure for **player modeling**: characterizing how individual players explore, exploit, and vary their strategies over the course of a game.

Gameplay data comes from [AiPerPacman](https://github.com/PipaFlores/Pacman-Unity_AiPerCog), an experimental, browser-playable reimplementation of the classic game built for this research. Each play session logs the full game state (Pacman and ghost positions, score, pill/power-pill events, etc.) at fixed time steps, alongside player metadata and psychometric survey data (BISBAS, flow).

This project is part of the [AiPerCog](https://www.helsinki.fi/en/researchgroups/high-performance-cognition/research) research project at the High Performance Cognition research group, University of Helsinki, which studies human gaming behavior and artificial intelligence modeling.

## Pipeline overview

The core of the project is `PatternAnalysis` ([src/analysis/pattern_analysis.py](src/analysis/pattern_analysis.py)), which orchestrates an end-to-end flow from raw game logs to player-modeling metrics. The pipeline is deliberately modular: each stage below is swappable, so the same flow can be run with different embedders, reducers, clusterers, or validation schemes to compare approaches.

```
raw game logs → slicing → embedding → dim. reduction → clustering → validation → latent-space analysis
```


### 1. Data loading and slicing

[`PacmanDataReader`](src/datahandlers/pacman_data_reader.py) reads the raw game/gamestate/user CSVs and produces [`Trajectory`](src/datahandlers/trajectory.py) objects (per-level coordinate + time series, with metadata such as user, session, duration, and outcome).

Because a full playthrough can be long and heterogeneous in length, the pipeline works over **sequence slices** rather than whole levels. Supported slicing modes (`sequence_type`) include:

- `first_5_seconds` / `last_5_seconds` — fixed-duration windows at the start/end of a level
- `whole_level` — the entire trajectory
- `pacman_attack` — windows centered on the periods where the player is actively hunting ghosts after a power pill, with configurable surrounding context
- `sliding_window` / `fixed_blocks` — regular re-chunking of a level into overlapping or non-overlapping blocks
- `first_50_steps` — a short fixed-length window, mainly used for fast debugging

A configurable feature set (`all_features` (244 feats), `Pacman` coordinates, ghost positions, ghost distances, score, or combinations thereof) is extracted and normalized (`global`, `sequence`, or `sample`-level normalization). Because this step can be slow at scale, results are content-fingerprinted and cached to disk (`cache/`) so repeated runs with the same configuration skip recomputation.

### 2. Embedding (representation learning)

Each sequence slice is reduced to a fixed-size latent vector, using one of two families of methods:

- **Deep autoencoders** (`src/models/`): all share a common [`BaseAutoencoder`](src/models/base.py) PyTorch Lightning interface trained on a masked reconstruction objective (so that padded/missing timesteps don't leak into the loss or the latent space). Supported architectures are `LSTM`, `MLP`, `Transformer` (a masked-imputation time-series transformer), `VAE`/`TimeVAE`, plus Keras/`aeon`-based convolutional and recurrent autoencoders (`DRNN`, `DCNN`, `ResNet`). Models are trained lazily, cached under `hpc/trained_models/`, and reused across runs unless `force_training=True`.
- **Geometric similarity** ([`GeomClustering`](src/analysis/geom_clustering.py)): skips learned embeddings entirely and instead clusters trajectories directly by pairwise similarity (DTW or Euclidean) over their raw (x, y) coordinates. This acts as a non-learned baseline for the deep-embedding approach.

- **Dim. red** Skips embedding phase entirely and instead just projects higher dimensionality of data into a lower dimensionality representation while preserving global/local structures, uses `UMAP` by default. This acts as another baseline alternative to for the deep-embedding approach.

### 3. Dimensionality reduction

When the embedding/latent dimension is larger than 2D, embeddings are projected down with `UMAP` (default) or `PCA` before clustering and visualization.

### 4. Clustering

Reduced embeddings (or, for geometric clustering, the trajectories' affinity matrix) are clustered with `HDBSCAN` (default) or `KMeans`, grouping gameplay slices into behaviorally similar clusters.

### 5. Validation

Cluster quality is assessed against an independent, literature-grounded reference: [**Behavlets**](src/analysis/behavlets.py), a rule-based behavioral pattern encoding scheme adapted from [Cowley & Charles (2016)](https://link.springer.com/article/10.1007/s11257-016-9170-1) — e.g. *"hunts close to the ghost house,"* *"times trapped by ghosts,"* *"ghost kileed."* *"avg. distance to ghosts"* Behavlet values are computed per sequence, recoded into discrete/binned labels where necessary, and compared against the learned cluster labels using **ARI**, **AMI**, **NMI**, and neighborhood-hit metrics ([`calculate_validation_measures`](src/analysis/pattern_analysis.py)). This quantifies whether the unsupervised structure discovered in latent space actually aligns with established, psychology-based player-behavior categories.

### 6. Latent-space / player-modeling analysis

[`latent_analysis.py`](src/analysis/latent_analysis.py) turns per-player, time-ordered sequences of latent points (or cluster labels) into player-modeling metrics, including:

- **Latent-path measures**: total/mean/std distance traveled through latent space, and distance from the population centroid — how much a player's strategy shifts over a session.
- **KDE-based novelty**: how unlikely a player's latent position is relative to the population density, both per-step and in aggregate (population-relative novelty).
- **Cluster-progression measures**: Shannon entropy, dominant-cluster fraction, and transition/repeat rates over a player's sequence of visited clusters — quantifying exploration vs. exploitation of strategies.
- **Recurrence Quantification Analysis (RQA)**: recurrence rate, laminarity, and trapping time of a trajectory through latent space (optional, requires `pyrqa`).

### 7. Visualization and summarization

`PatternAnalysis` and `src/visualization/` provide affinity-matrix overviews, static and interactive (Bokeh) latent-space plots colored by cluster/validation labels, cluster overviews, reconstruction-quality checks, and game replay/GIF rendering for qualitative inspection of individual clusters.

## Project structure

```
src/
├── analysis/         # PatternAnalysis pipeline, Behavlets, geometric clustering, latent-space metrics
├── datahandlers/      # PacmanDataReader, Trajectory, PyTorch Dataset/DataModule, feature normalization
├── models/            # Autoencoder architectures (LSTM, MLP, VAE, TimeVAE, transformer) sharing BaseAutoencoder
├── visualization/      # Game replay, trajectory, and cluster visualizers
├── utils/              # A* pathfinding, similarity measures, grid utilities, logging
└── tests/               # pytest suite, incl. image-baseline comparisons for visualizations

hpc/       # SLURM training/benchmarking scripts; validates model architectures against labeled
           # public time-series datasets (PenDigits, NATOPS, Worms, BasicMotions) independently
           # of the Pacman-specific pipeline. Also hosts trained_models/, affinity_matrices/,
           # and benchmark_results/ synced from the cluster (gitignored).
notebooks/ # Active notebooks demonstrating/using the pipeline (older, superseded notebooks
           # live under notebooks/Older_notebooks/)
data/      # Raw and processed datasets (not versioned; see rsync-*-excludes.txt)
claudio/   # Related sub-study: verbal fluency task foraging analysis
EDA/       # Exploratory data analysis (R)
cache/     # On-disk cache of pipeline intermediates (make_data, validation encodings)
```

## Installation

1. Clone the repository
2. Create the conda environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate pacman_encoder
```

The base environment covers geometric clustering, Behavlets, and the PyTorch-based autoencoders. The Keras/`aeon`-based architectures (`DRNN`, `DCNN`, `ResNet`) and `wandb` logging are optional and imported lazily — install `tensorflow`/`keras`/`wandb` separately if you need them. `RQA` metrics additionally require `pyrqa`.

## Usage
Minimal pipeline example:

```python
from src.analysis import PatternAnalysis

pipeline = PatternAnalysis(
    data_folder="../data",
    embedder="LSTM",
    sequence_type="first_5_seconds",
    validation_method="Behavlets",
)
pipeline.fit()
pipeline.summarize()
pipeline.plot_latent_space_overview()
```

Not maintained, notebooks live in `notebooks/`:

- [player_analysis.ipynb](notebooks/player_analysis.ipynb) — running the pipeline and inspecting player-level latent-space metrics
- [measures_latent_space.ipynb](notebooks/measures_latent_space.ipynb) — latent-space navigation/novelty measures in detail
- [Behavlet_extraction.ipynb](notebooks/Behavlet_extraction.ipynb) — computing Behavlet encodings used for validation
- [Behavioral_Foraging.ipynb](notebooks/Behavioral_Foraging.ipynb) / [Foraging_nature.ipynb](notebooks/Foraging_nature.ipynb) — foraging-behavior analyses

(Earlier notebooks covering the original trajectory-preprocessing/clustering/visualization workflow have been superseded by `PatternAnalysis` and moved to `notebooks/Older_notebooks/` for reference.)


## HPC and benchmarking

`hpc/` contains SLURM scripts for training embedding models at scale, and a separate benchmarking harness ([train_benchmark_autoencoders.py](hpc/train_benchmark_autoencoders.py)) that evaluates each autoencoder architecture on labeled public `aeon` time-series datasets. This mirrors the Pacman latent-space flow (embed → reduce → cluster) but against known ground-truth labels, scoring clusters with ARI/AMI/NMI to validate architecture choices independently of the (unlabeled) Pacman data.

## Testing

```bash
make test    # pytest -v
make lint    # ruff check --fix-only
make format  # ruff format
```

## Authors and Acknowledgment

Pablo Flores
High Performance Cognition Research Group, University of Helsinki.


[Behavlets: a method for practical player modelling using psychology-based player traits and domain specific features](https://link.springer.com/article/10.1007/s11257-016-9170-1)

[Utility of a Behavlets approach to a Decision theoretic predictive player model](https://arxiv.org/abs/1603.08973)

[Real-time rule-based classification of player types in computer games](https://link.springer.com/article/10.1007/s11257-012-9126-z)
