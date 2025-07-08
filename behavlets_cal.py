import multiprocessing as mp
from functools import partial
import pandas as pd
from src.analysis import BehavletsEncoding



if __name__ == "__main__":

    encoder = BehavletsEncoding(data_folder="data", verbose=True)

    encoder.calculate_all_levels_parallel(batch_size=2000, n_processes=6)

    print(encoder.summary_results.head())
    print(encoder.instance_details.head())