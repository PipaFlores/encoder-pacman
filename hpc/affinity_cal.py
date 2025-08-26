

import numpy as np
import time
import os
import multiprocessing
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.analysis import BehavletsEncoding, GeomClustering

def behavlet_affinity_calculation(BEHAV_TYPE):
    time_start = time.time()
    print("Initializing modules")
    Beh_encodings = BehavletsEncoding(data_folder="data/", verbose=True)
    clustering = GeomClustering(similarity_measure="dtw", verbose=True)

    t1 = time.time()
    print(
        f"Calculating behavlets for all levels (n = {len(Beh_encodings.reader.level_df)})"
    )
    for level_id in Beh_encodings.reader.level_df["level_id"]:
        Beh_encodings.calculate_behavlets(level_id=level_id, behavlet_type=BEHAV_TYPE)

    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

    print(f"Behavlets calculated in {time.time() - t1} seconds")
    t2 = time.time()
    print("Calculating affinity matrix")
    trajectories = Beh_encodings.get_trajectories(BEHAV_TYPE)
    print(f"Number of trajectories: {len(trajectories)}")
    clustering.calculate_affinity_matrix_parallel_cpu(
        trajectories=trajectories, n_jobs=None, chunk_size_multiplier=1
    )

    # clustering.calculate_affinity_matrix_parallel_cpu(trajectories=trajectories,
    #                                                   n_jobs=20,
    #                                                   chunk_size_multiplier=1)

    # clustering.calculate_affinity_matrix_parallel_cpu(trajectories=trajectories,
    #                                                   n_jobs=10,
    #                                                   chunk_size_multiplier=1)

    os.makedirs("affinity_matrices", exist_ok=True)
    np.savetxt(
        f"affinity_matrices/{BEHAV_TYPE}_dtw_affinity_matrix.csv",
        clustering.affinity_matrix,
        delimiter=",",
    )
    print(f"Affinity matrix calculated and saved in {time.time() - t2} seconds")

    print(f"Total time: {time.time() - time_start} seconds")

def calculate_beggining_of_games():
    import time
    from src.datahandlers import PacmanDataReader

    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    total_start = time.time()
    print("Initializing PacmanDataReader")
    reader_start = time.time()
    reader = PacmanDataReader(data_folder="../data/")
    print(f"PacmanDataReader initialized in {time.time() - reader_start:.2f} seconds")

    print("Initializing GeomClustering")
    clustering_start = time.time()
    clustering = GeomClustering(similarity_measure="dtw", verbose=True)
    print(f"GeomClustering initialized in {time.time() - clustering_start:.2f} seconds")

    traj_list = []

    print(f"Collecting first 200 steps of all games (n = {len(reader.level_df['level_id'].unique())})")
    collect_start = time.time()
    for level_id in reader.level_df["level_id"].unique():
        print(f"Getting partial trajectory for level_id: {level_id}")
        traj = reader.get_partial_trajectory(level_id, end_step=200)
        traj_list.append(traj)
    print(f"Collected {len(traj_list)} trajectories in {time.time() - collect_start:.2f} seconds")

    print(f"Number of trajectories collected: {len(traj_list)}")
    print("Calculating affinity matrix for first 200 steps of all games")
    affinity_start = time.time()
    clustering.calculate_affinity_matrix_parallel_cpu(
        trajectories=traj_list, n_jobs=None, chunk_size_multiplier=1
    )
    print(f"Affinity matrix calculated in {time.time() - affinity_start:.2f} seconds")

    os.makedirs("affinity_matrices", exist_ok=True)
    print("Saving affinity matrix to affinity_matrices/first_200_steps_dtw_affinity_matrix.csv")
    save_start = time.time()
    np.savetxt(
        f"affinity_matrices/first_200_steps_dtw_affinity_matrix.csv",
        clustering.affinity_matrix,
        delimiter=",",
    )
    print(f"Affinity matrix saved in {time.time() - save_start:.2f} seconds")
    print(f"Affinity matrix calculation and saving complete. Total time: {time.time() - total_start:.2f} seconds.")


if __name__ == "__main__":
    # behavlet_affinity_calculation("Aggression3")
    # behavlet_affinity_calculation("Aggression6")
    calculate_beggining_of_games()
