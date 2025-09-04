import pandas as pd
import os
import sys
import multiprocessing
from multiprocessing import Pool
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datahandlers import PacmanDataReader
from src.visualization import GameReplayer

# Global variables that will be accessible to worker processes
game_and_meta = None

def init_worker():
    """Initialize worker process with data"""
    global game_and_meta
    reader = PacmanDataReader(data_folder="../data")
    game_and_meta = pd.merge(
        reader.gamestate_df, reader.level_df, left_on="level_id", right_index=True
    )

def render_video_for_level(level_id):
    """Render video for a single level"""
    global game_and_meta
    
    video_path = f"videos/{level_id}.mp4"
    if not os.path.exists(video_path):
        try:
            print(f"Rendering video for level_id {level_id}...")
            level_gamestates = game_and_meta[game_and_meta["level_id"] == level_id]
            replayer = GameReplayer(data=level_gamestates, pathfinding=False)
            replayer.animate_session_compact(save_path=video_path, save_format="mp4")
        except Exception as e:
            print(f"Error when rendering level_id {level_id}: {e}")
            # Delete incomplete file if it exists
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Incomplete video {video_path} deleted.")
                except Exception as del_e:
                    print(f"Failed to delete incomplete video {video_path}: {del_e}")
    else:
        print(f"Video for {level_id} already exists, skipping.")

def parse_args():
    parser = argparse.ArgumentParser(description="Render video from data-logs")
    parser.add_argument('--test', action='store_true', help='run test (only 45 levels)')
    return parser.parse_args()

if __name__ == "__main__":
    # Create videos directory
    args = parse_args()
    os.makedirs("videos/", exist_ok=True)
    
    # Get level IDs
    reader = PacmanDataReader(data_folder="../data")
        
    level_ids = reader.level_df["level_id"].unique()
    if args.test:
        level_ids = level_ids[:45]
    
    N_JOBS = multiprocessing.cpu_count()
    
    # Use initializer to set up data in each worker process
    with Pool(N_JOBS, initializer=init_worker) as pool:
        pool.map(render_video_for_level, level_ids)





# for level_id in reader.level_df["level_id"].unique():
#     video_path = f"videos/{level_id}.mp4"
#     if not os.path.exists(video_path):
#         print(f"File videos/{level_id}.mp4 does not exist.")
#         level_gamestates = game_and_meta[game_and_meta["level_id"] == level_id]
#         replayer = GameReplayer(data=level_gamestates,
#                                 pathfinding=False)
        
#         replayer.animate_session_compact(save_path= video_path)
#     else:
#         print(f"video for {level_id} already exists, skipping")

