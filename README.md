# Encoder Pacman

This project contains scripts and notebooks for encoding and analyzing Pacman gameplay data. It provides tools for trajectory analysis, pattern mining, and visualization of Pacman gameplay patterns.

The data was collected through a web-based game, [AiPerPacman](https://github.com/PipaFlores/Pacman-Unity_AiPerCog), an experimental version of the classic game.


## Description

The project focuses on analyzing Pacman gameplay trajectories using various techniques including:
- Trajectory preprocessing and analysis
- Similarity measures for trajectory analysis
- Clustering of similar gameplay patterns
- Visualization of gameplay patterns and clusters in aggregated and non-aggregated forms
- Autoencoder-based representation learning

## Project Structure

- `notebooks/`: Contains Jupyter notebooks demonstrating key functionality
  - [Trajectories Preprocessing](notebooks/Trajectories_Preprocess.ipynb): Data preprocessing and trajectory analysis
  - [Pattern Mining](notebooks/PatternMining.ipynb): Discovering patterns in gameplay sequences
  - [Clustering with Similarity Measures](notebooks/Clustering_w_Similarity_Measures.ipynb): Clustering analysis of gameplay patterns
  - [Autoencoder](notebooks/Autoencoder.ipynb): Neural network-based representation learning
  - [Visualizations](notebooks/Visualizations.ipynb): Various visualization techniques for gameplay analysis

- `src/`: Source code for the project
- `data/`: Dataset storage (Currently not included in the repository)
- `environment.yml`: Conda environment specification

## Installation

1. Clone the repository
2. Create the conda environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate encoder-pacman
```

## Usage

The project's functionality is primarily demonstrated through Jupyter notebooks in the `notebooks/` directory. Each notebook focuses on a specific aspect of the analysis:

1. Start with [Trajectories Preprocessing](notebooks/Trajectories_Preprocess.ipynb) to understand the data preparation
2. Explore the main visualization methods in [Visualizations](notebooks/Visualizations.ipynb)
3. Check [Clustering with Similarity Measures](notebooks/Clustering_w_Similarity_Measures.ipynb) to find geometrical-based clustering analysis.

## Authors and Acknowledgment

Pablo Flores
High Performance Cognition research group, University of Helsinki.

This project is part of the [AiPerCog](https://www.helsinki.fi/en/researchgroups/high-performance-cognition/research) research project, which studies human gaming behavior and artificial intelligence modelling.

[Behavlets: a method for practical player modelling using psychology-based player traits and domain specific features](https://link.springer.com/article/10.1007/s11257-016-9170-1)

[Utility of a Behavlets approach to a Decision theoretic predictive player model](https://arxiv.org/abs/1603.08973)

[Real-time rule-based classification of player types in computer games](https://link.springer.com/article/10.1007/s11257-012-9126-z)

