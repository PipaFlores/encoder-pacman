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
    """Render video for a single level, unless a complete one already exists.

    Complete means one frame per game state. A render that gets interrupted can leave a video
    that plays but ends early - big enough to pass any size check - and it used to be skipped
    as done forever after, so the frame count is what decides. A new render goes to a temporary
    file and only replaces the real one once it has every frame.
    """
    global game_and_meta

    video_path = f"videos/{level_id}.mp4"
    tmp_path = f"videos/{level_id}.tmp.mp4"
    level_gamestates = game_and_meta[game_and_meta["level_id"] == level_id]
    expected_frames = len(level_gamestates)

    if os.path.exists(video_path):
        n_frames = GameReplayer.video_frame_count(video_path)
        if n_frames == expected_frames:
            print(f"Video for {level_id} already exists, skipping.")
            return
        print(f"Video for {level_id} has {n_frames} of {expected_frames} frames, re-rendering.")

    try:
        print(f"Rendering video for level_id {level_id}...")
        replayer = GameReplayer(data=level_gamestates, pathfinding=False)
        replayer.animate_session_compact(save_path=tmp_path, save_format="mp4")
        n_frames = GameReplayer.video_frame_count(tmp_path)
        if n_frames != expected_frames:
            print(f"Error: rendered video for level_id {level_id} has {n_frames} of {expected_frames} frames, discarded.")
        else:
            os.replace(tmp_path, video_path)
    except Exception as e:
        print(f"Error when rendering level_id {level_id}: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as del_e:
                print(f"Failed to delete incomplete video {tmp_path}: {del_e}")

def check_videos(reader, level_ids):
    """List levels whose video is missing or has fewer frames than the level has game states."""
    from concurrent.futures import ThreadPoolExecutor

    expected = reader.gamestate_df.groupby("level_id").size()
    paths = [f"videos/{level_id}.mp4" for level_id in level_ids]
    with ThreadPoolExecutor(8) as executor:
        frames = list(executor.map(
            lambda path: GameReplayer.video_frame_count(path) if os.path.exists(path) else None, paths
        ))
    incomplete = [(level_id, n) for level_id, n in zip(level_ids, frames) if n != expected[level_id]]
    print(f"{len(level_ids) - len(incomplete)} of {len(level_ids)} videos complete; "
          f"{len(incomplete)} missing or short (these get rendered on the next run):")
    for level_id, n in incomplete:
        print(f"  {level_id}: {n if n is not None else 'missing/unreadable'} of {expected[level_id]} frames")

def parse_args():
    parser = argparse.ArgumentParser(description="Render video from data-logs")
    parser.add_argument('--test', action='store_true', help='run test (only 45 levels)')
    parser.add_argument('--check', action='store_true', help='only list missing or incomplete videos, render nothing')
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

    if args.check:
        check_videos(reader, list(level_ids))
        sys.exit(0)
    
    # N_JOBS = multiprocessing.cpu_count()
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK",1))
    print(f"detected CPUs for multiprocess {N_JOBS}")  
    
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

