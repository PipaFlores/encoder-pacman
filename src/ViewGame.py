import argparse
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization.game_replayer import GameReplayer
from src.utils import PacmanDataReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pac-Man game states")
    parser.add_argument(
        "-g", "--game-id", type=int, default=None, help="Game ID to visualize"
    )
    parser.add_argument(
        "--playback-speed", type=float, default=1.0, help="Playback speed multiplier"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/", help="Path to the game state data"
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
    # parser.add_argument('--stats', type=str, default=None, help="Add stats columns to the stats panel")
    args = parser.parse_args()

    datareader = PacmanDataReader(data_folder=args.data_path, read_games_only=True)
    gamestate_df = datareader.gamestate_df
    game_df = datareader.game_df
    gamestate_df = pd.merge(gamestate_df, game_df, on="game_id", how="left")

    visualizer = GameReplayer(
        gamestate_df,
        game_id=args.game_id,
        playback_speed=args.playback_speed,
        verbose=args.verbose,
        pellets=args.pellets,
        pathfinding=args.pathfinding,
        #    stats_columns=args.stats.replace(" ", "").split(',') if args.stats else None,
        show_grid=args.grid,
    )
    visualizer.animate_session()
