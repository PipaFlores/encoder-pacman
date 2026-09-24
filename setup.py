from setuptools import setup, find_packages

setup(
    name="encoder-pacman",
    version="0.1.0",
    packages=find_packages(),
    # Runtime imports of src/ and hpc/. Optional extras (wandb, pacmap, pyrqa, aeon for the
    # benchmark, notebook tooling) and the ffmpeg binary are listed in environment.yml.
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "bokeh",
        "pillow",
        "tqdm",
        "torch",
        "pytorch-lightning",
        "scikit-learn",
        "umap-learn",
        "hdbscan",
        "pyyaml",
    ],
    python_requires=">=3.11",
)
