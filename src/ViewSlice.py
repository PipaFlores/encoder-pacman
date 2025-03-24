import argparse
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.game_replayer import GameReplayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pac-Man game states")
    parser.add_argument('-g', '--game-id', type=int, default=None, help="Game ID to visualize")
    parser.add_argument('--playback-speed', type=float, default=2.0, help="Playback speed multiplier")
    parser.add_argument('--data-path', type=str, default='data/gamestate.csv', help="Path to the game state data")
    parser.add_argument('--gamedata-path', type=str, default='data/game.csv', help="Path to the game data")
    parser.add_argument('--no-pellets', action='store_false', dest='pellets', help="Disable pellets")
    parser.add_argument('--no-pathfinding', action='store_false', dest='pathfinding', help="Disable A* path finding calculation")
    parser.add_argument('--verbose', action='store_true', default=False, help="Print verbose logging")
    parser.add_argument('--grid',action= 'store_true', default=False, help="Show grid")
    args = parser.parse_args()
    gamestate_df = pd.read_csv(args.data_path, converters={'user_id': lambda x: int(x)})

    game_df = gamestate_df.loc[gamestate_df['game_id'] == 601]
    game_df = game_df.loc[:, ['Pacman_X', 'Pacman_Y']]
    
    visualizer = GameReplayer(game_df, playback_speed=args.playback_speed, verbose=args.verbose)
    visualizer.animate_session()