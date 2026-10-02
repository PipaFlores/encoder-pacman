"""Render the per-sequence GIFs shown on hover in the augmented interactive plot.

Each GIF is cut with ffmpeg from its level's video in `videos/` (made by
video_rendering.py) and written to `gifs/` under the name
`PacmanDataReader.sequence_gif_names()` gives the sequence: `level_<id>_<start>_<end>.gif`,
positions within the level, end inclusive. That name depends only on which steps a
sequence covers - not on feature set, normalization or model - so one folder serves every
configuration, and `PatternAnalysis` (gifs_folder defaults to `<hpc_folder>/gifs`) finds
them without being told anything else.

A GIF is (re)rendered when it is missing or has fewer frames than steps, so a re-run renders
only what is missing or truncated, and an interrupted run never leaves a truncated GIF behind
under a real name. A GIF whose steps go past the end of its video is not attempted: that
video is incomplete (`video_rendering.py --check` lists them), and the affected levels are
reported so their videos can be fixed first.

    python render_gifs.py --sequence-type pacman_attack --context 20 --dry-run
    python render_gifs.py --config experiments/general_training.yaml
    sbatch render_gifs.sh --config experiments/general_training.yaml

With `--config`, the GIFs for every distinct slicing in the experiment are rendered, i.e.
what each of its configurations will want to show.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from PIL import Image

HPC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HPC_DIR, ".."))

sys.path.append(REPO_ROOT)

GIF_NAME = re.compile(r"level_(\d+)_(\d+)_(\d+)\.gif")

# Set per worker process by _init_worker.
_videos_folder = None
_gifs_folder = None
_replayer = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Experiment file; render for every distinct slicing it expands to.")
    parser.add_argument("--sequence-type", help="Slicing to render for (ignored with --config).")
    parser.add_argument("--context", type=int, default=20, help="pacman_attack context, as in the pipeline.")
    parser.add_argument("--filter-by-pill", type=int, default=None, help="pacman_attack pill filter (1-4).")
    parser.add_argument("--max-samples", type=int, default=None, help="Only the first N sequences, as in the pipeline.")
    parser.add_argument("--data-folder", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--videos-folder", default=os.path.join(HPC_DIR, "videos"))
    parser.add_argument("--gifs-folder", default=os.path.join(HPC_DIR, "gifs"))
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help="Parallel ffmpeg workers (default: SLURM_CPUS_PER_TASK, else 1).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count what would be rendered, render nothing.")
    args = parser.parse_args()

    if args.config is None and args.sequence_type is None:
        parser.error("give either --config or --sequence-type")
    return args


def slicings(args) -> list[dict]:
    """The distinct (sequence_type, context, filter_by_pill, max_samples) to render for."""
    if args.config is None:
        return [
            {
                "sequence_type": args.sequence_type,
                "context": args.context,
                "filter_by_pill": args.filter_by_pill,
                "max_samples": args.max_samples,
            }
        ]

    from experiment import expand, load_spec

    seen = []
    for config in expand(load_spec(args.config)):
        slicing = {
            "sequence_type": config.sequence_type,
            "context": config.context,
            "filter_by_pill": config.filter_by_pill,
            "max_samples": config.max_samples,
        }
        if slicing not in seen:
            seen.append(slicing)
    return seen


def gif_names_for(reader, slicing: dict) -> list[str]:
    """Same slicing and truncation as PacmanDataReader.make_data, without the featurizing."""
    raw_sequences = reader._slice_by_sequence_type(
        sequence_type=slicing["sequence_type"],
        context=slicing["context"],
        filter_by_pill=slicing["filter_by_pill"],
        rebase_scores=False,  # only the steps covered matter here
    )
    if slicing["max_samples"]:
        raw_sequences = raw_sequences[: slicing["max_samples"]]
    return reader.sequence_gif_names(raw_sequences)


def _init_worker(videos_folder: str, gifs_folder: str):
    global _videos_folder, _gifs_folder, _replayer
    from src.visualization import GameReplayer

    _videos_folder = videos_folder
    _gifs_folder = gifs_folder
    _replayer = GameReplayer()


def render_one(name: str) -> tuple[str, str]:
    """Render one GIF; returns (status, name) with status in ok / no_video / failed.
    (main() already filters out missing and incomplete videos; no_video is a last guard.)"""
    level_id, start, end = (int(group) for group in GIF_NAME.fullmatch(name).groups())
    video_path = os.path.join(_videos_folder, f"{level_id}.mp4")
    if not os.path.exists(video_path):
        return "no_video", name

    final_path = os.path.join(_gifs_folder, name)
    # Written under a temporary name and moved into place only once complete, so a killed
    # job can't leave a partial GIF that the next run would skip as already rendered.
    tmp_path = os.path.join(_gifs_folder, f"{name[:-4]}.tmp.gif")
    out = _replayer.extract_gamestate_subsequence_ffmpeg(
        video_path=video_path,
        start_gamestate=start,
        end_gamestate=end + 1,  # names are end-inclusive, ffmpeg's end is exclusive
        output_path=tmp_path,
    )
    if out is None or not os.path.exists(tmp_path):
        return "failed", name
    os.replace(tmp_path, final_path)
    return "ok", name


def main():
    args = parse_args()

    from src.datahandlers import PacmanDataReader

    reader = PacmanDataReader(data_folder=args.data_folder)

    wanted: list[str] = []
    for slicing in slicings(args):
        names = gif_names_for(reader, slicing)
        print(f"{slicing}: {len(names)} sequences")
        wanted.extend(names)
    wanted = sorted(set(wanted))

    def video_path(level: int) -> str:
        return os.path.join(args.videos_folder, f"{level}.mp4")

    def is_current(name: str) -> bool:
        # Complete = one frame per step; a GIF cut from an incomplete video has fewer. One short
        # is tolerated: GIFs rendered before end-inclusive names were cut one frame early.
        # (Reading the frame count is ~2 ms a GIF; file timestamps can't be trusted for this,
        # since copying or syncing videos resets theirs.)
        gif_path = os.path.join(args.gifs_folder, name)
        if not os.path.exists(gif_path):
            return False
        _, start, end = map(int, GIF_NAME.fullmatch(name).groups())
        try:
            with Image.open(gif_path) as gif:
                return gif.n_frames >= end - start
        except Exception:
            return False

    todo = [name for name in wanted if not is_current(name)]
    todo_levels = sorted({int(GIF_NAME.fullmatch(name).group(1)) for name in todo})
    no_video = [level for level in todo_levels if not os.path.exists(video_path(level))]

    from src.visualization import GameReplayer

    with_video = [level for level in todo_levels if level not in no_video]
    with ThreadPoolExecutor(max(args.jobs, 4)) as executor:
        frames = dict(zip(with_video, executor.map(GameReplayer.video_frame_count, map(video_path, with_video))))

    # A GIF needs frames start..end of its level's video; a video ending sooner is incomplete.
    short_video = sorted(
        {level for name in todo
         for level, _, end in [map(int, GIF_NAME.fullmatch(name).groups())]
         if level in frames and (frames[level] is None or end >= frames[level])}
    )
    skipped_levels = set(no_video) | set(short_video)
    n_needed = len(todo)
    todo = [name for name in todo if int(GIF_NAME.fullmatch(name).group(1)) not in skipped_levels]

    print(f"{len(wanted)} GIFs wanted: {len(wanted) - n_needed} up to date, {len(todo)} to render, "
          f"{n_needed - len(todo)} skipped (no complete video)")
    for levels, problem in ((no_video, "have no video"), (short_video, "have a video shorter than the level")):
        if levels:
            shown = ", ".join(str(level) for level in levels[:20])
            print(f"{len(levels)} levels {problem} in {args.videos_folder}; their GIFs are skipped until "
                  f"video_rendering.py (re)renders them: {shown}{' ...' if len(levels) > 20 else ''}")
    if args.dry_run or not todo:
        return

    os.makedirs(args.gifs_folder, exist_ok=True)
    print(f"Rendering with {args.jobs} worker(s)")
    started = time.perf_counter()
    counts = {"ok": 0, "no_video": 0, "failed": 0}
    failed = []
    with Pool(args.jobs, initializer=_init_worker, initargs=(args.videos_folder, args.gifs_folder)) as pool:
        for i, (status, name) in enumerate(pool.imap_unordered(render_one, todo, chunksize=8), start=1):
            counts[status] += 1
            if status == "failed":
                failed.append(name)
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.perf_counter() - started:.0f}s) {counts}", flush=True)

    if failed:
        print(f"ffmpeg failed for {len(failed)} GIFs, e.g. {', '.join(failed[:5])}")


if __name__ == "__main__":
    main()
