import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import src.visualization as vis
import src.utils as utils


visualizer = vis.game_visualizer(data_folder="C:\LocalData\pabflore\encoder-pacman\data", verbose=True)

# visualizer.trajectory_heatmap(game_id=601, normalize=True)

visualizer.trajectory_line_plot(game_id=601, time_step_delta=2, arrow_spacing=1)


_, _, _, _, gamedata, _  = utils.read_data('C:\LocalData\pabflore\encoder-pacman\data', game_list=[601])
game = gamedata[['movement_direction', 'input_direction', 'Pacman_X', 'Pacman_Y']].copy()

dx, dy = utils.calculate_velocities(game.loc[:, 'Pacman_X'].values, game.loc[:, 'Pacman_Y'].values)

direction_vector = pd.Series(list(zip(dx, dy)))

print(np.sign(dx), np.sign(dy))

