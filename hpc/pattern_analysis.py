import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.analysis import PatternAnalysis, GeomClustering




if __name__ == "__main__":
    pa = PatternAnalysis(
        data_folder= "../data",
        hpc_folder="./", # not necessary, but explicit
        sequence_type="last_5_seconds",
        embedder=None,
        clusterer = GeomClustering(similarity_measure="dtw", verbose=True),
        validation_method=None,
        using_hpc=True,
        verbose=True

    )
    pa.fit()
