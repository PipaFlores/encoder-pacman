# HPC experiments

Training runs of the `PatternAnalysis` pipeline on the cluster are driven from **one entry
point**, [`run_experiment.py`](run_experiment.py), which reads an *experiment file* — a YAML
description of a set of configurations to run — and executes them.

The former per-architecture scripts (`train_transformer.sh`, `train_LSTM.sh`,
`train_UMAP.sh`, `array_train.sh`) are superseded by experiment files in
[`experiments/`](experiments/); they duplicated the same SLURM header and bash loops, and
their sweeps were expanded by hand.

> The autoencoder benchmark harness (`train_benchmark_autoencoders.py` / `.sh`) is separate
> and unaffected. It evaluates architectures against labelled public `aeon` datasets rather
> than Pacman data.

## Quick start

```bash
# See what an experiment would run, without submitting anything
python run_experiment.py --config experiments/general_training.yaml --dry-run

# Submit it
./submit_experiment.sh experiments/general_training.yaml

# Run one configuration locally / on an interactive node
python run_experiment.py --config experiments/smoke.yaml --index 0
```

`--dry-run`, `--count` and `--slurm-flags` only expand the grid — they import nothing heavy
and are safe and fast on a login node.

## The pieces

| File | Role |
|---|---|
| [`experiments/*.yaml`](experiments/) | What to run. One file per experiment. |
| [`experiment.py`](experiment.py) | `RunConfig`, grid expansion, config hashing, and `run_one()` — the only code that touches `PatternAnalysis`. |
| [`run_experiment.py`](run_experiment.py) | The entry point: expands an experiment, runs configurations, writes results. |
| [`run_experiment.sh`](run_experiment.sh) | Generic SLURM job. One per experiment is never needed — the header is a fallback and gets overridden per submission. |
| [`submit_experiment.sh`](submit_experiment.sh) | Sizes the array, applies the experiment's `slurm:` block, picks the environment module, submits. |
| [`train_model.py`](train_model.py) | Single-configuration CLI for one-off and debugging runs. A thin front end over the same `run_one()`. |

## Experiment files

```yaml
name: general_training                 # names the output directory
description: "..."                     # becomes the wandb comment unless a config overrides it

slurm:                                 # rendered into sbatch flags at submit time
  partition: gpumedium
  account: project_2012947
  nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 72
  gres: "gpu:gh200:1"
  time: "04:00:00"
  use_array: false                     # optional, see "Submission modes"
  array_throttle: 2                    # optional, see "Submission modes"

defaults:                              # applied to every configuration
  sequence_type: pacman_attack
  feature_set: Experimental2
  embedder: LSTM

grid:                                  # cartesian product, in the key order written here
  latent_space: [32, 64, 128]
  sequence_type: [pacman_attack, last_5_seconds]

include:                               # explicit configurations, outside the product
  - {embedder: ResNet, latent_space: 128}

exclude:                               # drop configurations matching these (partial match)
  - {embedder: Transformer, latent_space: 2}
```

`grid` and `include` can be used together or separately. With only `include`, the
configurations are exactly those entries — `defaults` is a base for them, not a
configuration of its own.

**Every key under `defaults`, `grid` and `include` is a field of `RunConfig`**
([experiment.py](experiment.py)), which is also the list of `train_model.py` flags with
underscores. A typo fails at expansion time, naming the offender, rather than three hours
into a job. Booleans are written in the positive (`rebase_score: false`), even where the CLI
spells them as negations.

<details>
<summary>All configuration keys, with defaults</summary>

| Key | Default | | Key | Default |
|---|---|---|---|---|
| `sequence_type` | `first_5_seconds` | | `reducer` | `umap` |
| `context` | `20` | | `reducer_components` | `2` |
| `filter_by_pill` | `null` | | `umap_neighbors` | `15` |
| `rebase_score` | `true` | | `umap_min_dist` | `0.1` |
| `feature_set` | `Pacman` | | `umap_metric` | `euclidean` |
| `normalization` | `null` | | `clusterer` | `hdbscan` |
| `sort_ghost_distances` | `false` | | `min_cluster_size` | `20` |
| `embedder` | `LSTM` | | `min_samples` | `null` |
| `latent_space` | `256` | | `cluster_selection_epsilon` | `0.0` |
| `n_epochs` | `500` | | `kmeans_k` | `6` |
| `batch_size` | `32` | | `similarity_measure` | `euclidean` |
| `validation_split` | `0.3` | | `validation_method` | `Behavlets` |
| `dropout` | `0.1` | | `disable_wandb` | `false` |
| `elementwise_masking` | `false` | | `logging_comment` | `""` |
| `use_best` | `true` | | `test_dataset` | `false` |
| `max_samples` | `null` | | `test_run` | `false` |
| `seed` | `null` | | `verbose` | `false` |
| `ignore_cache` | `false` | | `data_folder` / `hpc_folder` / `cache_folder` | repo-relative |

</details>

### Available experiments

| File | What it is |
|---|---|
| [`smoke.yaml`](experiments/smoke.yaml) | 2 epochs on a handful of samples. Verifies plumbing, not results. Run this first after any change. |
| [`general_training.yaml`](experiments/general_training.yaml) | Every embedder once, each at its own latent size (an `include` list — edit a line to change one architecture's dimensionality). |
| [`embedder_sweep.yaml`](experiments/embedder_sweep.yaml) | Every embedder at a matched latent size, for comparing architectures. |
| [`latent_sweep_transformer.yaml`](experiments/latent_sweep_transformer.yaml) | Latent dimension × sequence type for the transformer (replaces `array_train.sh`). |
| [`kld_sweep.yaml`](experiments/kld_sweep.yaml) | KL weight × VAE/TimeVAE at a fixed latent size. The ladder is scaled to `seq_len × n_features`, so it is specific to that file's slicing and feature set. |
| [`transformer.yaml`](experiments/transformer.yaml) / [`lstm.yaml`](experiments/lstm.yaml) / [`umap_baseline.yaml`](experiments/umap_baseline.yaml) | The former single-architecture scripts. |

## Submission modes

`submit_experiment.sh` submits **one `sbatch` call** per experiment.

**Array mode (default).** One `sbatch --array=0-<N-1>`. SLURM then treats each task as an
independent job with its own JobID, node and GPU, and runs them in parallel as resources
allow. A 10-configuration sweep is 10 tasks running concurrently, not a loop.

**Sequential mode.** `use_array: false` in the `slurm:` block submits one ordinary job that
works through every configuration in order.

> **Why you would want this:** Slurm counts *every array task* as a separate job against the
> project's limit. A partition with a tight limit (`gputest`) will reject a 3-task array
> that would have accepted the same work as one job, with
> `Job violates accounting/QOS policy`. `smoke.yaml` uses `use_array: false` for exactly
> this reason. `array_throttle: N` is the milder option — it caps *concurrently running*
> tasks (`--array=0,1,2%N`) but does not reduce the number of submitted jobs, so it only
> helps against a running-job limit. Check which you are up against with:
> ```bash
> sacctmgr show qos format=name,maxjobspu,maxsubmitjobspu,maxjobspa,maxsubmitjobspa
> ```

## Output

```
runs/<experiment>/<run-id>/
├── <experiment>.yaml            # copy of what was submitted
├── configs.json                 # the full expansion, with index and config hash
├── results/<index>.json         # one file per configuration
├── results.csv / results.json   # the merge
└── <index>_<hash>/              # per configuration
    ├── config.json
    ├── latent_space_overview.png
    └── validation_measures.csv
```

The run id is fixed at submit time and shared by every job of the experiment, so a
two-module experiment still lands in one directory. It defaults to a timestamp;
`./submit_experiment.sh <config> <run-id>` reuses an existing one, which is how a group
whose `sbatch` failed gets re-submitted into the same place.

`results.csv` carries the measures plus the whole configuration as `cfg_*` columns, so it
can be read on its own. Configurations are identified by a **config hash** — a digest of the
configuration with paths and logging keys excluded, so the same experimental condition
hashes identically on the cluster and on a laptop.

### Collecting

Each configuration writes its own `results/<index>.json` rather than appending to a shared
CSV, because array tasks finish concurrently and would race each other's writes.

`run_experiment.py` merges them automatically **when one process ran every configuration** —
that is, a sequential submission with a single module group. Otherwise merge them once the
jobs have finished:

```bash
python run_experiment.py --config experiments/general_training.yaml --run-id <run-id> --collect
```

This also salvages a partial experiment: it merges whatever result files exist, so a job
killed at the walltime still yields a readable `results.csv` for the configurations that
completed.

### Weights & Biases

Runs are grouped by experiment (`WANDB_RUN_GROUP`) and typed by embedder
(`WANDB_JOB_TYPE`). Each run is named after its output directory —
`000_45c3de18_pacman_attack_Experimental2_MLP_h64` — so a run in the wandb UI leads straight
to its files on disk. Set `disable_wandb: true` to turn logging off.

## One-off runs

For a single configuration, `train_model.py` takes the same settings as command-line flags:

```bash
python train_model.py --embedder MLP --sequence-type pacman_attack \
    --feature-set Experimental2 --normalization global --latent-space 64 \
    --n-epochs 100 --verbose
```

It builds one `RunConfig` and calls the same `run_one()` the sweeps use — it does not
construct a pipeline of its own. That is deliberate: this CLI and the library had drifted
apart before (embedders reachable from `PatternAnalysis` but not from the command line,
pipeline arguments never exposed), and a second implementation is how that happens again.
`RunConfig.from_dict` rejects any flag that does not name a configuration field, so the
drift now fails at startup.

It writes no results directory. Use `run_experiment.py` for anything you want a record of.

## GIFs for the augmented interactive plot

`PatternAnalysis.plot_interactive_overview()` with `augmented_visualization=True` shows each
sequence's replay as a GIF on hover. The pipeline never renders those — it only looks them up
in `gifs/` (`gifs_folder`, default `<hpc_folder>/gifs`) and warns about missing ones.
`render_gifs.py` makes them, cutting each from its level's video in `videos/`:

```bash
python render_gifs.py --config experiments/general_training.yaml --dry-run  # count only
sbatch render_gifs.sh --config experiments/general_training.yaml            # SLURM
python render_gifs.py --sequence-type pacman_attack --context 20 --jobs 8   # locally
```

A GIF is named `level_<id>_<start>_<end>.gif`, the sequence's first and last step as
positions within its level (end inclusive), which depends only on the steps it covers. One
folder therefore serves every feature set, normalization and model, and `--config` renders
the union over an experiment's distinct slicings. A GIF that exists with one frame per step
is skipped, so a re-run (or a resubmission after a timeout) only renders what is missing or
truncated. ffmpeg and ffprobe must be on `PATH`.

A GIF can only be as complete as its level's video. Videos that end early — an interrupted
render, or a copy/sync that didn't finish, still plays fine — are detected by frame count
(one frame per game state): `render_gifs.py` skips and lists the affected levels, and

```bash
python video_rendering.py --check   # list missing or short videos, render nothing
```

lists them too; a normal `video_rendering.py` run re-renders exactly those.

`gifs/` is gitignored and excluded from the rsync-to-cluster list (like `videos/`); it comes
back with a from-cluster sync.

## Notes and gotchas

**Checkpoints are shared and every run trains from scratch.** Models land in
`trained_models/<sequence_type>/f<n_features>/<Class>_h<latent>_e<epochs>`, which does not
include the feature set *name*, normalization, context or masking — so configurations
differing only in those overwrite each other's weights. `force_training=True` means a stale
checkpoint is never loaded, so results are unaffected; the latent space and validation
measures are the deliverable, not the weights.

**Editing an experiment file renumbers its indices.** They are positions in the expansion.
Leave a running array alone until it finishes.

**Roihu memory.** Do not set `mem_per_cpu`. A reserved GH200 grants 217 GiB automatically,
and that is also the QOS ceiling — requesting memory explicitly can only overshoot it. Each
reserved GPU also grants 72 CPU cores, which cost nothing extra.

**Some architectures constrain `latent_space`.** The transformer uses it as `d_model` with
`n_heads=8`, so it must be a multiple of 8. `AEResNetClusterer` fixes its latent space at
128 regardless of what is requested. `TimeVAE` with `all_features` (267 columns) builds a
~17 GB decoder weight at `pacman_attack` sequence lengths — use a smaller feature set.

**`runs/` and `logs/` are gitignored**, along with `trained_models/`, `videos/` and `gifs/`,
and are excluded from the rsync-to-cluster list.
