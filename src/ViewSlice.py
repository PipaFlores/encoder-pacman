import argparse
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.game_replayer import GameReplayer
from src.datahandlers import PacmanDataReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pac-Man game states")
    parser.add_argument(
        "-g", "--game-id", type=int, default=None, help="Game ID to visualize"
    )
    parser.add_argument(
        "--playback-speed", type=float, default=2.0, help="Playback speed multiplier"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to the data",
    )
    parser.add_argument(
        "--gamedata-path",
        type=str,
        default="data/game.csv",
        help="Path to the game data",
    )
    parser.add_argument(
        "--no-pellets", action="store_false", dest="pellets", help="Disable pellets"
    )
    parser.add_argument(
        "--no-pathfinding",
        action="store_false",
        dest="pathfinding",
        help="Disable A* path finding calculation",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print verbose logging"
    )
    parser.add_argument("--grid", action="store_true", default=False, help="Show grid")
    args = parser.parse_args()

    data = PacmanDataReader(data_folder=args.data_path, read_games_only=True)

    game_and_meta = pd.merge(
        data.gamestate_df, data.level_df, left_on="level_id", right_index=True
    )

    slice = game_and_meta.loc[game_and_meta["level_id"] == 600]

    # game_df = game_df.loc[:, ["Pacman_X", "Pacman_Y"]

    visualizer = GameReplayer(
        slice, playback_speed=args.playback_speed, verbose=args.verbose
    )
    visualizer.animate_session(
        save_path="testanim_wtitle.mp4", save_format="mp4", title="testanim"
    )
