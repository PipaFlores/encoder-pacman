import os
import time
from typing import List

from src.datahandlers import PacmanDataReader
from src.visualization import GameReplayer
from src.utils import setup_logger

from .behavlets import Behavlets

# Initialize module-level logger
logger = setup_logger(__name__)


class BehavletsEncoding:
    """
    A class to perform the calculation, analysis and visualizations of Behavlets (Cowley & Charles, 2016)
    """

    def __init__(
        self, verbose: bool = False, debug: bool = False, data_folder="../data"
    ):
        """
        Initialize the Behavlets class
        """
        if debug:
            logger.setLevel("DEBUG")
        elif verbose:
            logger.setLevel("INFO")

        logger.info("Initializing BehavletsEncoding")
        self.reader = PacmanDataReader(data_folder=data_folder)
        self.behavlets = {
            name: Behavlets(name=name, verbose=verbose, debug=debug)
            for name in Behavlets.NAMES
        }

    def calculate_behavlets(
        self,
        level_id: int | None = None,
        user_id: int | None = None,
        behavlet_type: str = "all",
    ) -> List[Behavlets]:
        """
        Calculates behavlets for a level or game id
        """

        gamestates = self.reader._filter_gamestate_data(
            level_id=level_id, user_id=user_id
        )[0]

        for behavlet in self.behavlets.values():
            behavlet._reset_values()

        results = []

        if behavlet_type == "all":
            for behavlet in self.behavlets.values():
                results.append(behavlet.calculate(gamestates))
        else:
            results.append(self.behavlets[behavlet_type].calculate(gamestates))

        return results

    def behavlet_path_geom_clustering(self, behavlets: list[Behavlets]):
        raise NotImplementedError

    def create_replay(
        self,
        behavlet: Behavlets,
        instance: str | int | list[int] = "all",
        folder_path: str = "temp",
        save_format: str = "mp4",
        path_prefix: str = None,
        path_suffix: str = None,
        **kwargs,
    ):
        """
        Creates and stores a visualization of behavlets using GameReplayer.

        This method generates a video replay of the gameplay segment where a specific behavlet
        instance occurs. The replay is saved to the specified folder path in the given format.

        Args:
            behavlet (Behavlets): The behavlet instance to visualize
            instance (str | int): The specific instance/index, starting from 0, of the behavlet to visualize.
                If "all", visualizes all instances. Defaults to "all".

            folder_path (str): Path where the replay videos will be saved. Defaults to "temp".
            save_format (str): Format to save the video in (e.g., "mp4", "gif"). Defaults to "mp4".
            path_prefix (str): Optional prefix to add to the saved file name. Defaults to None.
            path_suffix (str): Optional suffix to add to the saved file name. Defaults to None.

        Returns:
            None: The method saves the visualization files but does not return anything.

        Note:
            - If the behavlet has a value of 0, no visualization is created
            - If the behavlet has no defined gameplay segments, no visualization is created
            - Each instance of the behavlet will be saved as a separate video file
        """
        if behavlet.value == 0:
            logger.info("Behavlet has a value of 0, no visualization created")
            return
        elif len(behavlet.gamesteps) == 0:
            logger.info(
                "Behavlet has non-zero value, but is not defined in any slice of gameplay, no visualization created"
            )
            return

        if instance == "all":
            instances = behavlet.gamesteps
            timesteps = behavlet.timesteps
            og_instance = list(range(len(behavlet.gamesteps)))
        elif isinstance(instance, int):
            instances = [behavlet.gamesteps[instance]]
            og_instance = [instance]
            timesteps = [behavlet.timesteps[instance]]
        elif isinstance(instance, list):
            if not isinstance(instance[0], int):
                raise ValueError(
                    f"Specific instances need to be given as a list of integers: {instance}"
                )
            if len(instance) > behavlet.instances:
                raise ValueError(
                    f"Requested more instances ({instance}) of {behavlet.name} than observed ({behavlet.instances})"
                )

            instances = []
            timesteps = []
            og_instance = instance
            for value in instance:
                instances.append(behavlet.gamesteps[value])
                timesteps.append(behavlet.timesteps[value])

        os.makedirs(folder_path, exist_ok=True)

        for idx, gamesteps in enumerate(instances):
            if gamesteps is not None:
                start_time = time.time()

                save_path = os.path.join(
                    folder_path,
                    f"{path_prefix or ''}{behavlet.name}_{og_instance[idx]}{path_suffix or ''}.{save_format}",
                )

                behavlet_slice = self.reader.gamestate_df.loc[
                    gamesteps[0] : gamesteps[1]
                ]

                replayer = GameReplayer(
                    data=behavlet_slice, pathfinding=kwargs.get("pathfinding", False)
                )

                animate_start = time.time()
                replayer.animate_session(
                    save_path=save_path,
                    title=f"Behavlet: {behavlet.full_name} instance {og_instance[idx]} at time {timesteps[idx]}",
                    save_format=save_format,
                )
                animate_time = time.time() - animate_start
                logger.info(f"Animation took {animate_time:.3f} seconds")

                total_time = time.time() - start_time
                logger.info(
                    f"Total processing time for instance {idx}: {total_time:.3f} seconds"
                )

        return
