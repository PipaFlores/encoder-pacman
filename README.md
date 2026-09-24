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

[`PacmanDataReader`](src/datahandlers/pacman_data_reader.py) reads the raw game/gamestate/user CSVs. The first time it runs, it preprocesses every game state (pellet-state maps, final-state and logging-bug fixes, and A\* maze-path distances from Pacman to each ghost) and caches the result in `data/gamestate.pkl`, which takes several minutes. Its `make_data()` method then slices levels into sequences and returns them, together with per-sequence metadata: user, session, level, duration and outcome, plus performance aggregates and psychometrics (flow, BISBAS).

Because a full playthrough can be long and heterogeneous in length, the pipeline works over **sequence slices** rather than whole levels. Supported slicing modes (`sequence_type`) include:

- `first_5_seconds` / `last_5_seconds` — fixed-duration windows at the start/end of a level
- `whole_level` — the entire trajectory
- `pacman_attack` — windows centered on the periods where the player is actively hunting ghosts after a power pill, with configurable surrounding context
- `sliding_window` / `fixed_blocks` — regular re-chunking of a level into overlapping or non-overlapping blocks
- `first_50_steps` — a short fixed-length window, mainly used for fast debugging

A configurable feature set (`feature_set`) is extracted from each slice:

- `Pacman`: Pacman's (x, y) coordinates
- `Pacman_Ghosts`: Pacman and ghost coordinates
- `Ghost_Distances`: the A\* distances to the 4 ghosts
- `Experimental2`: the ghost distances plus score
- `all_features`: 267 columns, i.e. 23 scalar features plus the 244-cell pellet-state map

Features are normalized at `global`, `sequence`, or `sample` level. Because this step can be slow at scale, results are content-fingerprinted and cached to disk (`cache/`) so repeated runs with the same configuration skip recomputation.

### 2. Embedding (representation learning)

Each sequence slice is reduced to a fixed-size latent vector, using one of two approaches:

- **Deep autoencoders** (`src/models/`): all share a common [`BaseAutoencoder`](src/models/base.py) PyTorch Lightning interface trained on a masked reconstruction objective (so that padded/missing timesteps don't leak into the loss or the latent space). Supported architectures are `LSTM`, `MLP`, `Transformer` (a masked-imputation time-series transformer), and `VAE`/`TimeVAE`. Models are trained lazily, cached under `hpc/trained_models/`, and reused across runs unless `force_training=True`.
- **Reducer only** (`embedder=None`): skips learned embeddings, and the reducer (`UMAP` by default) projects the flattened raw sequences directly to a low-dimensional representation. This acts as a non-learned baseline for the deep-embedding approach.

### 3. Dimensionality reduction

When the embedding/latent dimension is larger than 2D, embeddings are projected down with `UMAP` (default) or `PCA` before clustering and visualization.

### 4. Clustering

Reduced embeddings are clustered with `HDBSCAN` (default) or `KMeans`, grouping gameplay slices into behaviorally similar clusters.

### 5. Validation

Cluster quality is assessed against an independent, literature-grounded reference: [**Behavlets**](src/analysis/behavlets.py), a rule-based behavioral pattern encoding scheme adapted from [Cowley & Charles (2016)](https://link.springer.com/article/10.1007/s11257-016-9170-1).

The implemented behavlets are:

- Aggression 1, 3, 4 and 6: hunting close to the ghost house, ghost kills, hunting after the power pill wears off, and chasing ghosts vs. collecting pellets.
- Caution 1, 2a, 2b and 3: times trapped by ghosts, average distance to ghosts (overall and during hunts), and close calls.

Distance-based behavlets use the precomputed A\* maze-path distances, not Manhattan distance.

Behavlet values are computed per sequence and recoded into discrete/binned labels where necessary. Per-sequence metadata is added to the same validation set: level, duration, outcome and score change, per-user performance aggregates, and flow/BISBAS when the reader loads psychometrics. The whole set is compared against the learned cluster labels using **ARI**, **AMI**, **NMI**, and neighborhood-hit metrics ([`calculate_validation_measures`](src/analysis/pattern_analysis.py)). This quantifies whether the unsupervised structure discovered in latent space actually aligns with established, psychology-based player-behavior categories.

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
├── analysis/         # PatternAnalysis pipeline, Behavlets, latent-space metrics
├── datahandlers/      # PacmanDataReader, Trajectory, PyTorch Dataset/DataModule, feature normalization
├── models/            # Autoencoder architectures (LSTM, MLP, VAE, TimeVAE, transformer) sharing BaseAutoencoder
├── visualization/      # Game replay, trajectory, and cluster visualizers
├── utils/              # A* pathfinding, grid utilities, logging
└── tests/               # pytest suite, incl. image-baseline comparisons for visualizations

hpc/       # Experiment framework (YAML sweeps -> SLURM array jobs, see hpc/README.md), single-run
           # train_model.py, and the autoencoder benchmark on labeled public time-series datasets.
           # Also hosts trained_models/, benchmark_results/, runs/ and logs/ synced from the
           # cluster (gitignored).
notebooks/ # Analysis notebooks using the pipeline (kept locally, not versioned)
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

`environment.yml` covers `src/`, `hpc/` and the notebooks. `wandb` (logging), `pacmap` (reducer) and `pyrqa` (RQA metrics) are optional at runtime: they are imported lazily or only if installed, so they can be dropped from the file if not needed.

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

The analysis notebooks in `notebooks/` are kept out of version control, so they are not part of a fresh clone. They are not actively maintained, so some may lag behind the current API:

- `player_analysis.ipynb`: running the pipeline and inspecting player-level latent-space metrics
- `measures_latent_space.ipynb`: latent-space navigation/novelty measures in detail
- `Behavlet_extraction.ipynb`: computing the Behavlet encodings used for validation
- `Behavioral_Foraging.ipynb` / `Foraging_nature.ipynb`: foraging-behavior analyses


## HPC and benchmarking

`hpc/` has two parts.

**Experiment framework.** An experiment is a YAML file under [`hpc/experiments/`](hpc/experiments/) that gives defaults plus a grid of pipeline settings. [`run_experiment.py`](hpc/run_experiment.py) expands it into one hashed configuration per run, and [`submit_experiment.sh`](hpc/submit_experiment.sh) submits them as a SLURM array job, or as one sequential job. Results are collected under `hpc/runs/`. [`train_model.py`](hpc/train_model.py) runs a single configuration from command-line flags. See [hpc/README.md](hpc/README.md) for the configuration keys and submission modes.

**Autoencoder benchmark.** [train_benchmark_autoencoders.py](hpc/train_benchmark_autoencoders.py) evaluates each autoencoder architecture on labeled public time-series datasets, loaded through `aeon`. It mirrors the Pacman latent-space flow (embed → reduce → cluster), but against known ground-truth labels, scoring clusters with ARI/AMI/NMI to validate architecture choices independently of the (unlabeled) Pacman data.

## Testing

```bash
make test    # pytest -v
make lint    # ruff check --fix-only
make format  # ruff format
```

Most tests run against the real `data/` folder (they are integration tests rather than unit tests), so they need the dataset in place. The full suite takes several minutes, mostly spent loading data.

## Authors and Acknowledgment

Pablo Flores
High Performance Cognition Research Group, University of Helsinki.


[Behavlets: a method for practical player modelling using psychology-based player traits and domain specific features](https://link.springer.com/article/10.1007/s11257-016-9170-1)

[Utility of a Behavlets approach to a Decision theoretic predictive player model](https://arxiv.org/abs/1603.08973)

[Real-time rule-based classification of player types in computer games](https://link.springer.com/article/10.1007/s11257-012-9126-z)
