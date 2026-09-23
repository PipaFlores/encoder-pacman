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
    A class to perform the calculation and storage of results of Behavlets encodings (Cowley & Charles, 2016)
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

        # logger.info("Initializing BehavletsEncoding")
        self.data_folder = data_folder
        self.reader = PacmanDataReader(data_folder=data_folder)
        self.behavlets = {
            name: Behavlets(name=name, verbose=verbose, debug=debug)
            for name in Behavlets.NAMES
        }
        logger.debug(f"Behavlets initialized: {Behavlets.NAMES}")

    def calculate_behavlets_level(
        self,
        level_id: int,
        return_instance_details: bool = False, 
        behavlet_type: str | list[str] = "all",
    ):
        """
        Calculate Behavlets features for a given level.

        This method computes the specified Behavlets (or all, by default) for the provided level ID,
        using the loaded Pacman data. It returns summary results and, optionally, per-instance details.

        Args:
            level_id (int): The level ID for which to calculate Behavlets features.
            return_instance_details (bool, optional): If True, also return per-instance details for Behavlets
                that provide them (e.g., 'died', 'value_per_instance'). Defaults to False.
            behavlet_type (str or list of str, optional): The Behavlet(s) to calculate. If "all" (default),
                all available Behavlets are calculated. Otherwise, specify a single Behavlet name or a list of names.

        Returns:
            pd.DataFrame or tuple: If return_instance_details is False, returns a DataFrame of summary results.
                If True, returns a tuple (summary_results, instance_details).
        """

        gamestates, metadata = self.reader._filter_gamestate_data(
            level_id=level_id, include_metadata=True
        )

        results = []  # List of behavlet objects

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

        summary_results, instance_details = self._store_results(results, metadata)

        if return_instance_details:
            return summary_results, instance_details
        else:
            return summary_results

    def calculate_behavlets_gamestate_slice(self, 
                                            gamestates: pd.DataFrame,
                                            return_instance_details: bool = False, 
                                            behavlet_type: str | list[str] = "all"):

        results = []  # List of behavlet objects
        metadata = None
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

        summary_results, instance_details = self._store_results(results, metadata)

        if return_instance_details:
            return summary_results, instance_details
        else:
            return summary_results


    def _store_results(self, results: list["Behavlets"], metadata: pd.DataFrame | None = None):
        """Return behavlet results of a single level in a structured format as DataFrames (summary, instance details)"""

        # Prepare summary results (one row per level with all behavlets as columns)
        summary_data = {
            "level_id": metadata["level_id"] if metadata is not None else "",
            "user_id": metadata["user_id"] if metadata is not None else "",
        }

        # Add all behavlet metrics to a single row
        for behavlet in results:
            summary_data[f"{behavlet.name}_value"] = behavlet.value
            summary_data[f"{behavlet.name}_instances"] = behavlet.instances

            # # For inspection: get the gamesteps distance between behavlets
            # # Calculate the number of gamesteps between consecutive instances for all behavlets in this level
            # # (Note that this might return negative values for behavlets that are calculated per powerpill/quadrants)
            # gamesteps = summary_data.get(f"{behavlet.name}_gamesteps", [])
            # distances = []
            # if gamesteps and isinstance(gamesteps, list) and len(gamesteps) > 1:
            #     for i in range(len(gamesteps) - 1):
            #         current_end = gamesteps[i][1]
            #         next_start = gamesteps[i+1][0]
            #         if current_end is not None and next_start is not None:
            #             distances.append(next_start - current_end)
            # summary_data[f"{behavlet.name}_steps_between_instances"] = str(distances) if distances else None

            # Add behavlet-specific attributes based on output_attributes
            for attr in behavlet.output_attributes:
                if attr not in ["value", "instances"]:
                    summary_data[f"{behavlet.name}_{attr}"] = getattr(
                        behavlet, attr, None
                    )

        # Create summary row DataFrame
        summary_row = pd.DataFrame([summary_data])

        # Prepare instance details (one row per behavlet instance)
        instance_rows = []
        for behavlet in results:
            if len(behavlet.gamesteps) > 0:
                for i, (gamestep, timestep) in enumerate(
                    zip(behavlet.gamesteps, behavlet.timesteps)
                ):
                    if gamestep is not None:
                        instance_data = {
                            "level_id": metadata["level_id"] if metadata is not None else "",
                            "user_id": metadata["user_id"] if metadata is not None else "",
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

                        if filtered_instance_data:
                            instance_rows.append(filtered_instance_data)

        instance_details_df = pd.DataFrame(instance_rows) if instance_rows else pd.DataFrame()

        return summary_row, instance_details_df

    @staticmethod
    def get_vector_encodings(summary_results: pd.DataFrame) -> pd.DataFrame:
        """
        Get vector encodings from summary results, filtering for overall value columns.

        Args:
            summary_results (pd.DataFrame): Summary results as returned by
                `calculate_behavlets_level` / `calculate_behavlets_gamestate_slice`
                (or several of them concatenated).

        Returns:
            pd.DataFrame: Only the `<behavlet>_value` columns of `summary_results`.
        """
        value_columns = [col for col in summary_results.columns if col.endswith("_value")]
        return summary_results[value_columns]

    def get_trajectories(
        self,
        instance_details: pd.DataFrame,
        behavlet_name: str,
        level_id: int | None = None,
        extra_context: int | None = None,
    ):
        """
        Retrieve trajectories for a specified behavlet type.

        Parameters
        ----------
        instance_details : pd.DataFrame
            Per-instance details as returned by `calculate_behavlets_level(...,
            return_instance_details=True)` / `calculate_behavlets_gamestate_slice(...)`
            (or several of them concatenated).
        behavlet_name : str
            The name of the behavlet type for which to retrieve trajectories.
        level_id : int or None, optional
            If provided, only trajectories corresponding to this level are returned.
        extra_context : int or None, optional
            If provided, extends the start and end gamesteps of each trajectory by this value
            in both directions, within the bounds of the level's available gamesteps.

        Returns
        -------
        list or Trajectory
            A list of trajectory objects (or a single trajectory if only one is found),
            each annotated with relevant behavlet metadata.
        """

        # Filter instance_details for the given behavlet_name and (optionally) level_id
        df = instance_details[instance_details["behavlet_name"] == behavlet_name]
        if level_id is not None:
            df = df[df["level_id"] == level_id]

        trajectories = []

        for idx, row in df.iterrows():
            # Each row should have start_gamestep and end_gamestep
            start_gamestep = row.get("start_gamestep")
            end_gamestep = row.get("end_gamestep")
            if start_gamestep is None or end_gamestep is None:
                continue

            if extra_context:
                gamestates = self.reader.gamestate_df.loc[
                    self.reader.gamestate_df["level_id"] == row.get("level_id")
                ]
                first_state, last_state = (
                    gamestates.iloc[0].name,
                    gamestates.iloc[-1].name,
                )

                start_gamestep = max(first_state, start_gamestep - extra_context)
                end_gamestep = min(last_state, end_gamestep + extra_context)

            gamesteps = (start_gamestep, end_gamestep)
            trajectory = self.reader.get_trajectory(
                game_states=gamesteps, get_timevalues=True
            )
            trajectory.metadata["behavlet"] = behavlet_name
            # Add all output attributes from the behavlet to the metadata if present in row
            for output_attribute in self.behavlets[behavlet_name].output_attributes:
                if output_attribute in row:
                    trajectory.metadata[output_attribute] = row[output_attribute]
            # Add instance_idx if present
            if "instance_idx" in row:
                trajectory.metadata["instance_idx"] = row["instance_idx"]

            trajectory.metadata["gamesteps"] = gamesteps
            trajectory.metadata["timesteps"] = (
                row.get("start_timestep"),
                row.get("end_timestep"),
            )
            trajectories.append(trajectory)

        if len(trajectories) == 1:
            trajectories = trajectories[0]

        return trajectories


    def create_replay(
        self,
        instance_row: pd.Series,
        folder_path: str = "temp",
        save_format: str = "mp4",
        path_prefix: str = None,
        path_suffix: str = None,
        context_lenth: int = None,
        **kwargs,
    ):
        """
        Creates and stores a visualization of a behavlet instance using GameReplayer.

        This method generates a video replay of the gameplay segment where a specific behavlet
        instance occurs. The replay is saved to the specified folder path in the given format.

        Args:
            instance_row (pd.Series): A row from an instance_details dataframe containing
                the behavlet instance information including instance_idx, start_gamestep,
                end_gamestep, behavlet name, and other metadata.
            folder_path (str): Path where the replay videos will be saved. Defaults to "temp".
            save_format (str): Format to save the video in (e.g., "mp4", "gif"). Defaults to "mp4".
            path_prefix (str): Optional prefix to add to the saved file name. Defaults to None.
            path_suffix (str): Optional suffix to add to the saved file name. Defaults to None.
            context_length (int): Optional number of gamesteps to enlarge the visualization sequence.

        Returns:
            None: The method saves the visualization files but does not return anything.

        Note:
            - The method uses the instance_idx, start_gamestep, and end_gamestep from the instance_row
            - Each instance will be saved as a separate video file
        """
        # Extract information from the instance row
        instance_idx = instance_row["instance_idx"]
        start_gamestep = instance_row["start_gamestep"]
        end_gamestep = instance_row["end_gamestep"]
        behavlet_name = instance_row["behavlet_name"]

        if context_lenth:
            start_gamestep -= context_lenth
            end_gamestep += context_lenth

        # Get gamesteps as tuple
        gamesteps = (start_gamestep, end_gamestep)

        if gamesteps is None or start_gamestep is None or end_gamestep is None:
            logger.debug("Invalid gamesteps, no visualization created")
            return

        os.makedirs(folder_path, exist_ok=True)

        start_time = time.time()

        save_path = os.path.join(
            folder_path,
            f"{path_prefix or ''}{behavlet_name}_{instance_idx}{path_suffix or ''}.{save_format}",
        )

        behavlet_slice = self.reader.gamestate_df.loc[gamesteps[0] : gamesteps[1]]

        replayer = GameReplayer(
            data=behavlet_slice, pathfinding=kwargs.get("pathfinding", False)
        )

        animate_start = time.time()
        replayer.animate_session(
            save_path=save_path,
            title=f"Behavlet: {behavlet_name} instance {instance_idx}",
            save_format=save_format,
        )
        animate_time = time.time() - animate_start
        logger.debug(f"Animation took {animate_time:.3f} seconds")

        total_time = time.time() - start_time
        logger.debug(
            f"Total processing time for instance {instance_idx}: {total_time:.3f} seconds"
        )

        return
