"""Rebuild the latent spaces behind the results report and export what the report needs.

`run_one()` keeps only the validation measures and one overview PNG per configuration, so
once a sweep finishes the per-sequence latent coordinates, cluster labels and participant
ids are gone - and those are exactly what the cluster heatmaps (RQ1) and the participant
analyses (RQ2) are built from. This script rebuilds a few of a finished sweep's
configurations locally from their saved checkpoints and writes everything the report reads
into `<run>/report/`.

Nothing is trained here. Each configuration is rebuilt through the sweep's own
`build_analysis()` from its saved `config.json` (only the IO paths are swapped for local
ones), so the pipeline is the one the sweep ran, and `fit()` is refused if a checkpoint is
missing rather than silently retraining. Each rebuild is then checked against the sweep's
`validation_measures.csv`, and the result of that check lands in `manifest.json`.

    python hpc/build_report_data.py --run hpc/runs/general_sweep/20261012-170356
    python hpc/build_report_data.py --run ... --only f5s_lstm f5s_umap   # redo a subset
    python hpc/build_report_data.py --run ... --site                     # and build the page

Outputs (under <run>/report/):
    manifest.json                         selections, reproduction checks, KDE/RQA parameters
    rq1/validation_long.csv               every sweep run x validation set (the comparison table)
    rq1/rebuilt_<key>_validation.csv      the same measures for each local rebuild
    rq1/attack_rebuilds.csv               all attack rebuilds (trained, untrained, no encoder) together
    rq1/probes/<key>_{hits,ridge}.csv     full-dimension vs 2-D probes of each attack space (report_probes.py)
    latent/<key>_sequences.csv            per-sequence latent coordinates, cluster, metadata
    figures/<key>_*.png                   latent space and validation panels
    clusters/<key>/cluster_NN.png         per-cluster overview (velocity grid, heatmap, samples)
    rq2/<key>_participant_measures.csv    continuous / cluster / RQA measures per participant
    rq2/<key>_sequence_series.csv         per-step novelty along each participant's trajectory
    rq2/<key>_correlations.csv            measures x performance/psychometrics, Pearson
    rq2/participant_stats.csv             performance and psychometrics per participant
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiment import HPC_DIR, RAW_EMBEDDERS, REPO_ROOT, RunConfig, build_analysis
from report_probes import probe_space

from src.analysis.latent_analysis import (
    Recurrence_Quantification_Analysis,
    fit_global_step_novelty_transform,
    heuristic_bandwidth,
    population_relative_novelty,
    step_novelty,
    summarize_cluster_progression,
    summarize_continuous_progression,
)

warnings.filterwarnings("ignore", message=".*non-interactive.*")

# The pacman_attack constructs the report leads with (see the RQ1 section).
FOCUS_BEHAVLETS = [
    "Aggression3_value",
    "Aggression6_value",
    "Caution2a_value",
    "Caution2b_value",
]

# Which sweep configurations to rebuild. Matched on these config fields only, so the
# no-embedder baselines match whatever latent_space default they were expanded with.
SELECTIONS = {
    "f5s_lstm": {
        "sequence_type": "first_5_seconds",
        "feature_set": "all_features",
        "embedder": "LSTM",
        "latent_space": 512,
    },
    "f5s_umap": {
        "sequence_type": "first_5_seconds",
        "feature_set": "all_features",
        "embedder": None,
    },
    "pa_lstm": {
        "sequence_type": "pacman_attack",
        "feature_set": "Experimental2",
        "embedder": "LSTM",
        "latent_space": 512,
    },
    "pa_mlp": {
        "sequence_type": "pacman_attack",
        "feature_set": "Experimental2",
        "embedder": "MLP",
        "latent_space": 512,
    },
    "pa_vae": {
        "sequence_type": "pacman_attack",
        "feature_set": "Experimental2",
        "embedder": "VAE",
        "latent_space": 512,
    },
    "pa_transformer": {
        "sequence_type": "pacman_attack",
        "feature_set": "Experimental2",
        "embedder": "Transformer",
        "latent_space": 512,
    },
    "pa_umap": {
        "sequence_type": "pacman_attack",
        "feature_set": "Experimental2",
        "embedder": None,
    },
}
# Local-only spaces: a sweep configuration rebuilt with overrides, so they have no sweep
# counterpart. Untrained controls embed with the encoder's initial weights (whatever the
# trained space adds over its control is what training bought); "Masked UMAP" is the
# no-encoder baseline with the padding-aware metric (src/utils/masked_distance.py).
VARIANTS = {
    **{
        f"{key}_untrained": (key, {"test_initialized_model": True})
        for key in ("pa_lstm", "pa_mlp", "pa_vae", "pa_transformer")
    },
    "pa_masked_umap": ("pa_umap", {"embedder": "Masked UMAP"}),
}
ATTACK = "pacman_attack"
# What the probes predict and score: the focus Behavlets, plus sequence length to show
# whether a space carries the length leak.
PROBE_TARGETS = [*FOCUS_BEHAVLETS, "sequence_length"]
# Participant (RQ2) analyses and cluster heatmaps run on these; validation panels on the rest.
RQ2_KEYS = ("f5s_lstm", "f5s_umap")

# Same choices as the Behavioral_Foraging notebook: participants need a minimum history for
# the per-participant measures to mean anything, and the RQA neighbourhood is a fixed share
# of the latent space's extent.
MIN_SEQUENCES = 10
RQA_RADIUS_FRACTION = 0.1
NOVELTY_MIN_HISTORY = 2

RQA_KEYS = ("Recurrence_Rate", "Laminarity", "Trapping_Time", "V_max")

# Participant-level measures that enter the correlation table. cluster_repeat_fraction is
# left out: it is exactly 1 - cluster_transition_rate.
MEASURES = [
    "n_sequences",
    "latent_path_mean_distance",
    "kde_step_novelty_mean",
    "kde_pop_relative_novelty_mean",
    "visited_clusters",
    "Shannon_entropy_normalized",
    "cluster_dominant_fraction",
    "cluster_noise_fraction",
    "cluster_transition_rate",
    "Recurrence_Rate",
    "Laminarity",
    "Trapping_Time",
    "V_max",
]
EXTERNAL = [
    "max_score",
    "mean_score",
    "score_per_time",
    "max_level_reached",
    "win_ratio",
    "mean_flow",
    "BIS",
    "REW",
    "DRIVE",
    "FUN",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------------------
# Sweep bookkeeping
# --------------------------------------------------------------------------------------


def run_dirs(run_root: Path) -> list[Path]:
    return sorted(p for p in run_root.iterdir() if p.is_dir() and p.name[:3].isdigit())


def find_run(run_root: Path, **criteria) -> Path:
    matches = []
    for run_dir in run_dirs(run_root):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if all(config.get(key) == value for key, value in criteria.items()):
            matches.append(run_dir)
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one run matching {criteria}, found {[m.name for m in matches]}"
        )
    return matches[0]


def validation_long(run_root: Path) -> pd.DataFrame:
    """Every run's per-validation-set measures in one long table, with its configuration."""
    frames = []
    for run_dir in run_dirs(run_root):
        measures_path = run_dir / "validation_measures.csv"
        if not measures_path.exists():
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        measures = pd.read_csv(measures_path)
        measures.insert(0, "run", run_dir.name)
        measures.insert(1, "sequence_type", config["sequence_type"])
        measures.insert(2, "feature_set", config["feature_set"])
        measures.insert(3, "embedder", config["embedder"] or "UMAP-only")
        measures.insert(
            4, "latent_space", config["latent_space"] if config["embedder"] else np.nan
        )
        frames.append(measures)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Rebuilding a configuration
# --------------------------------------------------------------------------------------


def local_config(run_dir: Path, **overrides) -> RunConfig:
    """The run's own configuration with the cluster's IO paths swapped for local ones."""
    values = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    values.update(
        data_folder=os.path.join(REPO_ROOT, "data"),
        hpc_folder=HPC_DIR,
        cache_folder=os.path.join(REPO_ROOT, "cache"),
        disable_wandb=True,
        verbose=False,
        **overrides,
    )
    return RunConfig.from_dict(values)


def rebuild(config: RunConfig):
    analysis = build_analysis(config)

    def refuse_training(*args, **kwargs):
        raise RuntimeError(
            f"No checkpoint at {analysis.model_path}: refusing to retrain here, since a fresh "
            "model would not be the one the sweep evaluated."
        )

    analysis._train_model = refuse_training
    analysis.fit(force_training=False)
    return analysis


def reproduction_check(analysis, run_dir: Path) -> dict:
    """How far the rebuilt latent space is from the one the sweep evaluated."""
    swept = pd.read_csv(run_dir / "validation_measures.csv").set_index("validation_set")
    rebuilt = analysis.validation_measures.set_index("validation_set")
    shared = swept.index.intersection(rebuilt.index)
    diff = (
        swept.loc[shared, ["neigh_hit_adj", "NMI"]]
        - rebuilt.loc[shared, ["neigh_hit_adj", "NMI"]]
    ).abs()
    result = json.loads(
        (run_dir.parent / "results" / f"{int(run_dir.name[:3])}.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "validation_sets_compared": len(shared),
        "max_abs_diff_neigh_hit_adj": float(diff["neigh_hit_adj"].max()),
        "max_abs_diff_NMI": float(diff["NMI"].max()),
        "n_clusters_sweep": result.get("n_clusters"),
        "n_clusters_rebuilt": len(set(analysis.labels) - {-1}),
        "n_samples": len(analysis.reduced_embeddings),
    }


def sequence_frame(analysis) -> pd.DataFrame:
    """One row per sequence: latent position, cluster and every 1-D metadata field."""
    n = len(analysis.reduced_embeddings)
    frame = pd.DataFrame(
        {
            "x": analysis.reduced_embeddings[:, 0],
            "y": analysis.reduced_embeddings[:, 1],
            "cluster": analysis.labels,
        }
    )
    for key, values in analysis.metadata_dictionary.items():
        values = np.asarray(values)
        if values.ndim == 1 and len(values) == n and key not in frame:
            frame[key] = values
    if analysis.validation_encodings is not None:
        for column in FOCUS_BEHAVLETS:
            if column in analysis.validation_encodings:
                frame[column] = analysis.validation_encodings[column].to_numpy()
                frame[f"{column}_label"] = analysis.validation_labels[column].to_numpy()
    return frame


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def save(fig, path: Path, dpi: int = 110) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def space_title(config: RunConfig) -> str:
    """Figure title naming the latent space, e.g. 'pacman_attack | Experimental2 | LSTM h512'."""
    if config.embedder is None:
        embedder = "UMAP-only"
    elif config.embedder in RAW_EMBEDDERS:
        embedder = config.embedder
    else:
        embedder = f"{config.embedder} h{config.latent_space}"
    if config.test_initialized_model:
        embedder += " (untrained)"
    return f"{config.sequence_type} | {config.feature_set} | {embedder}"


def space_kind(config: RunConfig) -> str:
    """ "trained", "untrained" (initial weights) or "baseline" (no encoder)."""
    if config.test_initialized_model:
        return "untrained"
    if config.embedder is None or config.embedder in RAW_EMBEDDERS:
        return "baseline"
    return "trained"


def write_probes(analysis, config: RunConfig, key: str, out: Path) -> None:
    """Neighbourhood-hit and ridge probes of one attack space (see report_probes.py)."""
    if config.embedder == "Masked UMAP":
        kind = "masked_umap"
    elif config.embedder is None or config.embedder in RAW_EMBEDDERS:
        kind = "plain_umap"
    else:
        kind = "latent"
    log(f"    probes ({kind})")
    hits, ridges = probe_space(analysis, config.umap_metric, kind, PROBE_TARGETS)
    probe_dir = out / "rq1" / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    hits.to_csv(probe_dir / f"{key}_hits.csv", index=False)
    ridges.to_csv(probe_dir / f"{key}_ridge.csv", index=False)
    if kind == "latent":
        # Kept so later probes can run without rebuilding the space (row order as in
        # latent/<key>_sequences.csv when that exists).
        np.save(
            out / "latent" / f"{key}_embeddings.npy",
            np.asarray(analysis.embeddings, dtype=np.float32),
        )


def plot_focus_panels(analysis, title: str):
    """The latent space coloured by each of the four focus Behavlets."""
    return analysis.plot_latent_space_overview(
        validation_set=FOCUS_BEHAVLETS,
        black_background=True,
        colormap="viridis",
        dotsize=0.2,
        pretty_val_title=True,
        all_labels_in_legend=True,
        custom_title=title,
    )


def plot_cluster_heatmaps(analysis, title: str):
    """Every cluster's normalised visit heatmap, one panel per cluster."""
    from src.visualization.game_visualizer import GameVisualizer

    labels = np.asarray(analysis.labels)
    clusters = sorted(set(labels) - {-1})

    # Panels large enough that the routes stay readable under the maze-wall overlay.
    visualizer = GameVisualizer(darkmode=True)
    ncols = 5
    nrows = math.ceil(len(clusters) / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.6, nrows * 4.2), squeeze=False
    )
    for ax, cluster in zip(axs.flat, clusters):
        trajectories = [
            t for t, label in zip(analysis.trajectory_list, labels) if label == cluster
        ]
        visualizer.plot_heatmap(trajectory=trajectories, normalize=True, ax=ax)
        ax.set_title(
            f"Cluster {cluster} (n={len(trajectories)})", color="white", fontsize=12
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.flat[len(clusters) :]:
        ax.set_visible(False)
    fig.suptitle(title, color="white", fontsize=14)
    fig.tight_layout()
    return fig


def cluster_figures(analysis, key: str, title: str, out: Path) -> pd.DataFrame:
    """Latent space by cluster, one overview per cluster, and all cluster heatmaps on one page."""
    labels = np.asarray(analysis.labels)
    clusters = sorted(set(labels) - {-1})

    fig = analysis.plot_latent_space_overview(dotsize=1, custom_title=title)
    save(fig, out / "figures" / f"{key}_clusters.png")

    for cluster in clusters:
        analysis.plot_cluster_overview(cluster_id=cluster, darkmode=True)
        save(plt.gcf(), out / "clusters" / key / f"cluster_{cluster:02d}.png", dpi=72)

    save(
        plot_cluster_heatmaps(analysis, title),
        out / "figures" / f"{key}_cluster_heatmaps.png",
    )

    embedded = analysis.reduced_embeddings
    return pd.DataFrame(
        [
            {
                "cluster": cluster,
                "size": int((labels == cluster).sum()),
                "x": float(embedded[labels == cluster, 0].mean()),
                "y": float(embedded[labels == cluster, 1].mean()),
            }
            for cluster in clusters
        ]
    )


# --------------------------------------------------------------------------------------
# RQ2: participant dynamics
# --------------------------------------------------------------------------------------


def participant_order(metadata: dict) -> dict:
    """Each participant's sequence indices in the order they were played."""
    users = np.asarray(metadata["user_id"])
    progression = np.asarray(metadata["total_levels_played"], dtype=float)
    order = {}
    for user in np.unique(users):
        indices = np.flatnonzero(users == user)
        order[user] = indices[np.argsort(progression[indices], kind="stable")]
    return order


def participant_analysis(analysis) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from scipy.spatial.distance import pdist

    embedded = np.asarray(analysis.reduced_embeddings)
    labels = np.asarray(analysis.labels)
    order = participant_order(analysis.metadata_dictionary)

    bandwidth = heuristic_bandwidth(embedded, method="scott")
    rqa_epsilon = float(pdist(embedded).max() * RQA_RADIUS_FRACTION)
    # Step novelty is calibrated across participants, so one participant's values are
    # comparable with another's (see fit_global_step_novelty_transform).
    calibration = fit_global_step_novelty_transform(
        [embedded[indices] for indices in order.values()],
        min_history=NOVELTY_MIN_HISTORY,
        bandwidth=bandwidth,
    )
    population_novelty = population_relative_novelty(
        embedded, embedded, bandwidth=bandwidth, normalize=True
    )

    measures, series = [], []
    for user, indices in order.items():
        points = embedded[indices]
        row = {"user_id": user}
        row |= summarize_continuous_progression(
            points,
            embedded,
            bandwidth=bandwidth,
            min_history=NOVELTY_MIN_HISTORY,
            global_step_normalizer=calibration,
        )
        row |= summarize_cluster_progression(labels[indices])
        # Recurrence structure needs a history to recur over, and only participants above
        # MIN_SEQUENCES are reported, so the rest are not run through pyrqa at all.
        if len(points) >= MIN_SEQUENCES:
            row |= Recurrence_Quantification_Analysis(points, epsilon=rqa_epsilon)
        else:
            row |= {key: np.nan for key in RQA_KEYS}
        measures.append(row)

        steps = step_novelty(
            points,
            min_history=NOVELTY_MIN_HISTORY,
            bandwidth=bandwidth,
            global_normalizer=calibration,
        )
        for step, (index, novelty) in enumerate(zip(indices, steps)):
            series.append(
                {
                    "user_id": user,
                    "step": step,
                    "sequence_index": int(index),
                    "x": float(points[step, 0]),
                    "y": float(points[step, 1]),
                    "cluster": int(labels[index]),
                    "step_novelty": float(novelty),
                    "pop_novelty": float(population_novelty[index]),
                }
            )

    parameters = {
        "kde_bandwidth": bandwidth,
        "kde_bandwidth_rule": "heuristic_bandwidth(method='scott', scale='mean_std')",
        "rqa_epsilon": rqa_epsilon,
        "rqa_radius_fraction": RQA_RADIUS_FRACTION,
        "novelty_min_history": NOVELTY_MIN_HISTORY,
        "min_sequences": MIN_SEQUENCES,
    }
    return pd.DataFrame(measures), pd.DataFrame(series), parameters


def participant_stats(data_folder: str) -> pd.DataFrame:
    """Performance (all levels a participant played) and psychometrics, one row per participant.

    Needs the reader's psychological tables, which only exist when it was loaded with
    read_games_only=False. The reader is a singleton, so this upgrades the instance the
    rebuilds used; that re-reads the data and is why it runs last.
    """
    from src.datahandlers import PacmanDataReader

    reader = PacmanDataReader(data_folder=data_folder, read_games_only=False)

    rows = []
    for user in reader.level_df["user_id"].unique():
        levels = reader.level_df[reader.level_df["user_id"] == user]
        bisbas = reader.bisbas_df[reader.bisbas_df["user_id"] == user]
        bisbas = bisbas.iloc[0] if len(bisbas) == 1 else None
        rows.append(
            {
                "user_id": user,
                "max_score": levels["max_score"].max(),
                "mean_score": levels["max_score"].mean(),
                "score_per_time": (levels["max_score"] / levels["duration"]).mean(),
                "max_level_reached": levels["level"].max(),
                "win_ratio": levels["win"].mean(),
                "mean_flow": reader.game_flow_df.loc[
                    reader.game_flow_df["user_id"] == user, "FLOW"
                ].mean(),
                **{
                    key: (bisbas[key] if bisbas is not None else np.nan)
                    for key in ("BIS", "REW", "DRIVE", "FUN")
                },
            }
        )
    return pd.DataFrame(rows)


def correlations(measures: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Pearson r for every pair among the measures and the participant variables.

    Measure x participant-variable pairs are the tested family, so their Bonferroni p is
    corrected over that family; the remaining pairs are descriptive.
    """
    from scipy.stats import pearsonr

    merged = measures[measures["n_sequences"] >= MIN_SEQUENCES].merge(
        stats, on="user_id", how="left"
    )
    variables = MEASURES + EXTERNAL
    n_tests = len(MEASURES) * len(EXTERNAL)

    rows = []
    for i, a in enumerate(variables):
        for b in variables[i + 1 :]:
            pair = merged[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(pair[a], pair[b])
            tested = (a in MEASURES) != (b in MEASURES)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "r": r,
                    "p": p,
                    "n": len(pair),
                    "tested": tested,
                    "p_bonferroni": min(1.0, p * n_tests)
                    if tested and not np.isnan(p)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run",
        required=True,
        help="A finished run directory, e.g. hpc/runs/general_sweep/<run-id>.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[*SELECTIONS, *VARIANTS],
        help="Rebuild only these spaces and keep everything else already in <run>/report/.",
    )
    parser.add_argument(
        "--site",
        action="store_true",
        help="Also build the report page from the result (hpc/build_report_site.py).",
    )
    args = parser.parse_args()
    keys = args.only or [*SELECTIONS, *VARIANTS]

    # Figures are only ever saved here, so plt.show() inside the pipeline must not block.
    # Switched here rather than at import so the notebooks can import these helpers.
    plt.switch_backend("Agg")

    run_root = Path(args.run).resolve()
    out = run_root / "report"
    for sub in ("rq1", "rq2", "latent", "figures", "clusters"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    manifest = {
        "run": str(run_root),
        "focus_behavlets": FOCUS_BEHAVLETS,
        "selections": {},
    }
    if args.only and manifest_path.exists():
        manifest["selections"] = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "selections"
        ]

    log("Collecting the sweep's validation measures")
    validation_long(run_root).to_csv(out / "rq1" / "validation_long.csv", index=False)

    for key, criteria in SELECTIONS.items():
        if key not in keys:
            continue
        run_dir = find_run(run_root, **criteria)
        config = local_config(run_dir)
        log(f"{key}: rebuilding {run_dir.name} ({config.label()})")
        started = time.perf_counter()
        analysis = rebuild(config)
        check = reproduction_check(analysis, run_dir)
        log(f"    reproduction: {check}")

        entry = {
            "run_dir": run_dir.name,
            "label": config.label(),
            "config_hash": config.config_hash(),
            "reproduction": check,
        }
        title = space_title(config)
        sequence_frame(analysis).to_csv(
            out / "latent" / f"{key}_sequences.csv", index=False
        )
        analysis.validation_measures.to_csv(
            out / "rq1" / f"rebuilt_{key}_validation.csv", index=False
        )
        shutil.copy(
            run_dir / "latent_space_overview.png",
            out / "figures" / f"{key}_validation_grid.png",
        )

        if key in RQ2_KEYS:
            log("    cluster figures")
            cluster_figures(analysis, key, title, out).to_csv(
                out / "latent" / f"{key}_clusters.csv", index=False
            )
            log("    participant measures")
            measures, series, parameters = participant_analysis(analysis)
            measures.to_csv(
                out / "rq2" / f"{key}_participant_measures.csv", index=False
            )
            series.to_csv(out / "rq2" / f"{key}_sequence_series.csv", index=False)
            entry["rq2_parameters"] = parameters
            entry["n_participants"] = len(measures)
            entry["n_participants_min_sequences"] = int(
                (measures["n_sequences"] >= MIN_SEQUENCES).sum()
            )
        else:
            save(
                plot_focus_panels(analysis, title),
                out / "figures" / f"{key}_focus_behavlets.png",
            )
        if config.sequence_type == ATTACK:
            write_probes(analysis, config, key, out)

        entry["seconds"] = round(time.perf_counter() - started, 1)
        manifest["selections"][key] = entry
        del analysis
        plt.close("all")

    for key, (base_key, overrides) in VARIANTS.items():
        if key not in keys:
            continue
        run_dir = find_run(run_root, **SELECTIONS[base_key])
        config = local_config(run_dir, **overrides)
        title = space_title(config)
        log(f"{key}: rebuilding {run_dir.name} with {overrides}")
        analysis = rebuild(config)
        save(
            plot_focus_panels(analysis, title),
            out / "figures" / f"{key}_focus_behavlets.png",
        )
        if space_kind(config) == "baseline":
            # No sweep run to copy the full validation grid from, so draw it here.
            save(
                analysis.plot_latent_space_overview(
                    validation_set="all",
                    black_background=True,
                    colormap="viridis",
                    dotsize=0.2,
                    pretty_val_title=True,
                    custom_title=title,
                ),
                out / "figures" / f"{key}_validation_grid.png",
                dpi=80,
            )
        analysis.validation_measures.to_csv(
            out / "rq1" / f"rebuilt_{key}_validation.csv", index=False
        )
        write_probes(analysis, config, key, out)
        manifest["selections"][key] = {
            "run_dir": run_dir.name,
            "label": config.label(),
            "overrides": overrides,
            "test_initialized_model": bool(config.test_initialized_model),
        }
        del analysis
        plt.close("all")

    # Every rebuilt attack space in one table, so trained, untrained and no-encoder spaces
    # are compared on numbers from the same machine.
    attack_rebuilds = []
    for key, entry in manifest["selections"].items():
        path = out / "rq1" / f"rebuilt_{key}_validation.csv"
        base_key, overrides = VARIANTS.get(key, (key, {}))
        if not path.exists() or base_key not in SELECTIONS:
            continue
        config = local_config(
            run_root / entry["run_dir"], **overrides
        )  # the space as rebuilt
        if config.sequence_type == ATTACK:
            attack_rebuilds.append(
                pd.read_csv(path).assign(
                    key=key,
                    embedder=config.embedder or "UMAP-only",
                    kind=space_kind(config),
                )
            )
    if attack_rebuilds:
        pd.concat(attack_rebuilds, ignore_index=True).to_csv(
            out / "rq1" / "attack_rebuilds.csv", index=False
        )

    rebuilt_rq2 = [key for key in RQ2_KEYS if key in keys]
    if rebuilt_rq2:
        stats_path = out / "rq2" / "participant_stats.csv"
        if args.only and stats_path.exists():
            stats = pd.read_csv(stats_path)  # independent of the latent space
        else:
            log("Participant performance and psychometrics (re-reads the data)")
            stats = participant_stats(os.path.join(REPO_ROOT, "data"))
            stats.to_csv(stats_path, index=False)
        for key in rebuilt_rq2:
            measures = pd.read_csv(out / "rq2" / f"{key}_participant_measures.csv")
            correlations(measures, stats).to_csv(
                out / "rq2" / f"{key}_correlations.csv", index=False
            )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log(f"Done. Report data in {out}")

    if args.site:
        from build_report_site import build_site

        build_site(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
