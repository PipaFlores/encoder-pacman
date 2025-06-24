import os
import time
from typing import List
import pandas as pd

from src.datahandlers import PacmanDataReader
from src.visualization import GameReplayer
from src.utils import setup_logger

from .behavlets import Behavlets

# Initialize module-level logger
logger = setup_logger(__name__)


class BehavletsEncoding:
    """
    A class to perform the calculation, analysis and visualizations of Behavlets (Cowley & Charles, 2016)

    TODO:
    - Data structure for results
      - All attributes, full with none values it doesnt matter

    - Trajectory extractions of behavlets (Gett all trajectories where X behavlet happens)
    - Aggregate visualizations (Heatmaps and velocity grids)
    - Maybe avoid clustering procedure in this script.

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
        logger.info(f"Behavlets initialized: {Behavlets.NAMES}")

        self.summary_results = pd.DataFrame()
        self.instance_details = pd.DataFrame()
        self.special_attributes = pd.DataFrame()

    def calculate_behavlets(
        self,
        level_id: int,
        behavlet_type: str | list[str] = "all",
    ) -> List[Behavlets]:
        """
        Calculates behavlets for a level and updates the self.summary_results, self.instance_details and self.special_attributes.

        Args:
            level_id: The level id to calculate the behavlets for.
            behavlet_type: The type of behavlet to calculate. If "all", all behavlets are calculated.

        Returns:
            None
        """

        fitered_data = self.reader._filter_gamestate_data(
            level_id=level_id, include_metadata=True
        )
        gamestates = fitered_data[0]
        metadata = fitered_data[1]

        for behavlet in self.behavlets.values():
            behavlet._reset_values()

        results = []

        if behavlet_type == "all":
            for behavlet_name in self.behavlets.keys():
                results.append(self.behavlets[behavlet_name].calculate(gamestates))
        elif isinstance(behavlet_type, list):
            for behavlet_name in behavlet_type:
                if behavlet_name not in self.behavlets:
                    raise ValueError(f"Invalid behavlet name: {behavlet_name}")
                results.append(self.behavlets[behavlet_name].calculate(gamestates))
        elif isinstance(behavlet_type, str):
            if behavlet_type not in self.behavlets:
                raise ValueError(f"Invalid behavlet name: {behavlet_type}")
            results.append(self.behavlets[behavlet_type].calculate(gamestates))
        else:
            raise ValueError(f"Invalid behavlet type: {behavlet_type}")

        self._store_results(results, metadata)

    def _store_results(self, results: list[Behavlets], metadata: pd.DataFrame):
        """Store behavlet results of a single level in a structured format"""

        ## Store summary results (one row per level with all behavlets as columns)
        summary_data = {
            "level_id": metadata["level_id"],
            "user_id": metadata["user_id"],
        }

        # Add all behavlet metrics to a single row
        for behavlet in results:
            summary_data[f"{behavlet.name}_value"] = behavlet.value
            summary_data[f"{behavlet.name}_instances"] = behavlet.instances

            # Add behavlet-specific attributes based on output_attributes
            for attr in behavlet.output_attributes:
                if attr not in ["value", "instances"]:
                    summary_data[f"{behavlet.name}_{attr}"] = getattr(
                        behavlet, attr, None
                    )

        # Create summary row and append to results
        summary_row = pd.DataFrame([summary_data], index=[metadata["level_id"]])
        self.summary_results = pd.concat(
            [self.summary_results, summary_row], ignore_index=False
        )

        ## Store instance details (one row per behavlet instance)
        for behavlet in results:
            # Handle multiple instances per behavlet
            if len(behavlet.gamesteps) > 0:
                for i, (gamestep, timestep) in enumerate(
                    zip(behavlet.gamesteps, behavlet.timesteps)
                ):
                    if gamestep is not None:  # Skip None entries
                        instance_data = {
                            "level_id": metadata["level_id"],
                            "user_id": metadata["user_id"],
                            "behavlet_name": behavlet.name,
                            "instance_idx": i,
                            "start_gamestep": gamestep[0]
                            if isinstance(gamestep, tuple)
                            else gamestep,
                            "end_gamestep": gamestep[1]
                            if isinstance(gamestep, tuple)
                            else gamestep,
                            "start_timestep": timestep[0]
                            if isinstance(timestep, tuple)
                            else timestep,
                            "end_timestep": timestep[1]
                            if isinstance(timestep, tuple)
                            else timestep,
                            "instant_gamestep": behavlet.instant_gamestep[i]
                            if i < len(behavlet.instant_gamestep)
                            else None,
                            "instant_position": behavlet.instant_position[i]
                            if i < len(behavlet.instant_position)
                            else None,
                            "value_per_instance": behavlet.value_per_instance[i]
                            if i < len(behavlet.value_per_instance)
                            else None,
                            "value_per_pill": behavlet.value_per_pill[i]
                            if i < len(behavlet.value_per_pill)
                            else None,
                        }
                        for attr in behavlet.output_attributes:
                            if attr not in [
                                "value",
                                "instances",
                                "gamesteps",
                                "timesteps",
                                "value_per_instance",
                                "instant_gamestep",
                                "instant_position",
                                "value_per_pill",
                            ]:
                                attr_value = getattr(behavlet, attr, None)
                                if (
                                    attr_value is not None
                                    and isinstance(attr_value, list)
                                    and i < len(attr_value)
                                ):
                                    instance_data[attr] = attr_value[i]
                                else:
                                    instance_data[attr] = None

                        # Filter out None values to avoid FutureWarning
                        filtered_instance_data = {
                            k: v for k, v in instance_data.items() if v is not None
                        }

                        if (
                            filtered_instance_data
                        ):  # Only create DataFrame if we have non-None data
                            instance_row = pd.DataFrame([filtered_instance_data])
                            self.instance_details = pd.concat(
                                [self.instance_details, instance_row], ignore_index=True
                            )

    def get_trajectories(self, behavlet_name: str, level_id: int | None = None):
        """Get trajectories for a behavlet"""
        if level_id is None:
            level_ids = self.summary_results["level_id"].unique()
        else:
            level_ids = [level_id]

        trajectories = []

        for level_id in level_ids:
            gamesteps = self.summary_results.loc[level_id, f"{behavlet_name}_gamesteps"]
            if gamesteps is None:
                continue
            if isinstance(gamesteps, tuple):
                gamesteps = [gamesteps]
                timesteps = [
                    self.summary_results.loc[level_id, f"{behavlet_name}_timesteps"]
                ]
                trajectory = self.reader.get_trajectory(
                    game_states=gamesteps, get_timevalues=True
                )
                trajectory.metadata["behavlet"] = f"{behavlet_name}"
                trajectories.append(trajectory)
                for output_attribute in self.behavlets[behavlet_name].output_attributes:
                    trajectory.metadata[output_attribute] = self.summary_results.loc[
                        level_id, f"{behavlet_name}_{output_attribute}"
                    ]
                    trajectory.metadata["instance_idx"] = 0

            elif isinstance(gamesteps, list):
                for idx, gamestep in enumerate(gamesteps):
                    trajectory = self.reader.get_trajectory(
                        game_states=gamestep, get_timevalues=True
                    )
                    trajectory.metadata["behavlet"] = f"{behavlet_name}"
                    for output_attribute in self.behavlets[
                        behavlet_name
                    ].output_attributes:
                        trajectory.metadata[output_attribute] = (
                            self.summary_results.loc[
                                level_id, f"{behavlet_name}_{output_attribute}"
                            ]
                        )
                        trajectory.metadata["instance_idx"] = idx
                    trajectories.append(trajectory)
            else:
                raise ValueError(f"Invalid type for gamesteps: {type(gamesteps)}")

        return trajectories

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
            logger.debug("Behavlet has a value of 0, no visualization created")
            return
        elif len(behavlet.gamesteps) == 0:
            logger.debug(
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
                logger.debug(f"Animation took {animate_time:.3f} seconds")

                total_time = time.time() - start_time
                logger.debug(
                    f"Total processing time for instance {idx}: {total_time:.3f} seconds"
                )

        return
