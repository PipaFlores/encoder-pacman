import os
import time
from typing import List
import pandas as pd
import multiprocessing as mp
from functools import partial

from src.datahandlers import PacmanDataReader
from src.visualization import GameReplayer
from src.utils import setup_logger

from .behavlets import Behavlets

# Initialize module-level logger
logger = setup_logger(__name__)


def _calculate_single_level(level_id, data_folder, behavlet_types, verbose=False, debug=False):
    """
    Calculate behavlets for a single level in isolation.
    This function needs to be at module level to be pickleable for multiprocessing.
    
    Args:
        level_id: The level id to calculate behavlets for
        data_folder: Path to the data folder
        behavlet_types: Which behavlets to calculate ("all", str, or list[str])
        verbose: Enable verbose logging
        debug: Enable debug logging
        
    Returns:
        dict: Contains level_id and the three result dataframes
    """
    try:
        # Each process gets its own BehavletsEncoding instance
        encoder = BehavletsEncoding(
            data_folder=data_folder, 
            verbose=verbose, 
            debug=debug
        )
        
        # Calculate behavlets for this level
        encoder.calculate_behavlets(level_id=level_id, behavlet_type=behavlet_types)
        
        # Return the results instead of storing in instance variables
        return {
            'level_id': level_id,
            'summary_results': encoder.summary_results,
            'instance_details': encoder.instance_details,
            'special_attributes': encoder.special_attributes,
            'success': True,
            'error': None
        }
    except Exception as e:
        # Return error information if something goes wrong
        logger.error(f"Error processing level {level_id}: {str(e)}")
        return {
            'level_id': level_id,
            'summary_results': pd.DataFrame(),
            'instance_details': pd.DataFrame(),
            'special_attributes': pd.DataFrame(),
            'success': False,
            'error': str(e)
        }


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

        logger.info("Initializing BehavletsEncoding")
        self.data_folder = data_folder  # Store data_folder for parallel processing
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

        gamestates, metadata = self.reader._filter_gamestate_data(
            level_id=level_id, include_metadata=True
        )


        results = [] # List of behavlet objects

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

    def calculate_behavlets_category(self, level_id, behavlet_category):
        """
        Calculates behavlets of a specific category for a single level
        """

        gamestates, metadata = self.reader._filter_gamestate_data(
            level_id=level_id, include_metadata=True
        )

        results = []

        for behavlet_name in self.behavlets.keys():
            if self.behavlets[behavlet_name].category == behavlet_category:
                results.append(self.behavlets[behavlet_name].calculate(gamestates))
        
        self._store_results(results, metadata)


    def calculate_behavlets_parallel(
        self, 
        level_ids: list[int], 
        behavlet_type: str | list[str] = "all",
        n_processes: int = None,
        verbose: bool = False,
        debug: bool = False
    ):
        """
        Calculate behavlets for multiple levels in parallel using multiprocessing.
        
        Args:
            level_ids: List of level IDs to process
            behavlet_type: Which behavlets to calculate ("all", str, or list[str])
            n_processes: Number of processes to use (defaults to min(cpu_count, len(level_ids)))
            verbose: Enable verbose logging for worker processes
            debug: Enable debug logging for worker processes
            
        Returns:
            None: Results are stored in self.summary_results, self.instance_details, self.special_attributes
        """
        if not level_ids:
            logger.warning("No level IDs provided for parallel calculation")
            return
            
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(level_ids))
        
        logger.info(f"Starting parallel calculation for {len(level_ids)} levels using {n_processes} processes")
        start_time = time.time()
        
        # Create partial function with fixed arguments
        calc_func = partial(
            _calculate_single_level,
            data_folder=self.data_folder,
            behavlet_types=behavlet_type,
            verbose=verbose,
            debug=debug
        )
        
        # Process levels in parallel
        try:
            with mp.Pool(n_processes) as pool:
                results = pool.map(calc_func, level_ids)
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            raise
        
        # Track successful and failed calculations
        successful_results = []
        failed_levels = []
        
        for result in results:
            if result['success']:
                successful_results.append(result)
            else:
                failed_levels.append((result['level_id'], result['error']))
        
        # Report any failures
        if failed_levels:
            logger.warning(f"Failed to process {len(failed_levels)} levels:")
            for level_id, error in failed_levels:
                logger.warning(f"  Level {level_id}: {error}")
        
        # Merge successful results
        logger.info(f"Merging results from {len(successful_results)} successful calculations")
        
        for result in successful_results:
            if not result['summary_results'].empty:
                self.summary_results = pd.concat(
                    [self.summary_results, result['summary_results']], 
                    ignore_index=False
                )
            
            if not result['instance_details'].empty:
                self.instance_details = pd.concat(
                    [self.instance_details, result['instance_details']], 
                    ignore_index=True
                )
            
            if not result['special_attributes'].empty:
                self.special_attributes = pd.concat(
                    [self.special_attributes, result['special_attributes']], 
                    ignore_index=True
                )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel calculation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully processed {len(successful_results)}/{len(level_ids)} levels")

    def calculate_all_levels_parallel(
        self, 
        behavlet_type: str | list[str] = "all",
        n_processes: int = None,
        batch_size: int = 100,
        verbose: bool = False,
        debug: bool = False
    ):
        """
        Calculate behavlets for all available levels in parallel, processing in batches.
        
        Args:
            behavlet_type: Which behavlets to calculate ("all", str, or list[str])
            n_processes: Number of processes to use
            batch_size: Number of levels to process in each batch (helps manage memory)
            verbose: Enable verbose logging for worker processes
            debug: Enable debug logging for worker processes
        """
        all_level_ids = self.reader.level_df["level_id"].tolist()
        logger.info(f"Processing {len(all_level_ids)} levels in batches of {batch_size}")
        
        # Process in batches to manage memory
        for i in range(0, len(all_level_ids), batch_size):
            batch_ids = all_level_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_level_ids) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} levels)")
            
            self.calculate_behavlets_parallel(
                level_ids=batch_ids,
                behavlet_type=behavlet_type,
                n_processes=n_processes,
                verbose=verbose,
                debug=debug
            )

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

    def get_vector_encodings(self) -> pd.DataFrame:
        """Get vector encodings from summary results, filtering for overall value columns."""
        # Get all rows from summary_results
        all_rows = self.summary_results
        
        # Filter columns to only include those ending with '_value'
        value_columns = [col for col in all_rows.columns if col.endswith('_value')]
        
        # Return the filtered DataFrame with only value columns
        return all_rows[value_columns]

    def get_trajectories(self, behavlet_name: str, level_id: int | None = None, extra_context: int | None = None):
        """
        Retrieve trajectories for a specified behavlet type.

        Parameters
        ----------
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
        df = self.instance_details
        df = df[df["behavlet_name"] == behavlet_name]
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
                gamestates = self.reader.gamestate_df.loc[self.reader.gamestate_df["level_id"] == row.get("level_id")]
                first_state , last_state = gamestates.iloc[0].name, gamestates.iloc[-1].name

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
            trajectory.metadata["timesteps"] = (row.get("start_timestep") , row.get("end_timestep"))
            trajectories.append(trajectory)

        if len(trajectories) == 1:
            trajectories = trajectories[0] 

        return trajectories

        return trajectories

    def behavlet_path_geom_clustering(self, behavlets: list[Behavlets]):
        raise NotImplementedError

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
            instance_row (pd.Series): A row from self.instance_details dataframe containing
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

        behavlet_slice = self.reader.gamestate_df.loc[
            gamesteps[0] : gamesteps[1]
        ]

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

