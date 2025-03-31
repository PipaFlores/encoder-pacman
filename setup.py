from setuptools import setup, find_packages

setup(
    name="encoder-pacman",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "ipympl",
        "seaborn",
        "stumpy",
        "lightning",
        "scikit-learn",
    ],
    python_requires=">=3.11",
) 