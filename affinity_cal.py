from src.analysis import BehavletsEncoding, GeomClustering
from src.datahandlers import PacmanDataReader
import numpy as np
import time
import os
import multiprocessing


def affinity_calculation(BEHAV_TYPE):

    time_start = time.time()
    print("Initializing modules")
    Beh_encodings = BehavletsEncoding(data_folder="data/", verbose = True)
    clustering = GeomClustering(similarity_measure="dtw", verbose = True)

    t1 = time.time()
    print(f"Calculating behavlets for all levels (n = {len(Beh_encodings.reader.level_df)})")
    for level_id in Beh_encodings.reader.level_df["level_id"]:
        Beh_encodings.calculate_behavlets(level_id=level_id, behavlet_type=BEHAV_TYPE)

    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

    print(f"Behavlets calculated in {time.time() - t1} seconds")
    t2 = time.time()
    print("Calculating affinity matrix")
    trajectories = Beh_encodings.get_trajectories(BEHAV_TYPE)
    print(f"Number of trajectories: {len(trajectories)}")
    clustering.calculate_affinity_matrix_parallel_cpu(trajectories=trajectories,
                                                      n_jobs=None,
                                                      chunk_size_multiplier=1)
    
    # clustering.calculate_affinity_matrix_parallel_cpu(trajectories=trajectories,
    #                                                   n_jobs=20,
    #                                                   chunk_size_multiplier=1)
    
    # clustering.calculate_affinity_matrix_parallel_cpu(trajectories=trajectories,
    #                                                   n_jobs=10,
    #                                                   chunk_size_multiplier=1)
    

    os.makedirs(f"affinity_matrices", exist_ok=True)
    np.savetxt(f"affinity_matrices/{BEHAV_TYPE}_dtw_affinity_matrix.csv", clustering.affinity_matrix, delimiter=",")
    print(f"Affinity matrix calculated and saved in {time.time() - t2} seconds")

    print(f"Total time: {time.time() - time_start} seconds")

if __name__ == "__main__":
    affinity_calculation("Aggression3")
    affinity_calculation("Aggression6")








