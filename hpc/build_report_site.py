"""Build the results report page from a run's report/ folder.

`build_report_data.py` writes the numbers and figures into <run>/report/; this turns them
into one self-contained page:

    python hpc/build_report_site.py --run hpc/runs/general_sweep/20261012-170356
    python hpc/build_report_data.py --run ... --site     # rebuild the data, then this

Output: <run>/report/site/report.html (data inlined) and <run>/report/site/fig/ (web JPGs).

The page is hpc/report_site/template.html (layout and interaction, driven entirely by the
data) plus hpc/report_site/narrative.html (the interpretation, written by hand for one run).
When the narrative was written for another run than the one being built, the page carries
a visible warning, so old text never sits silently next to new numbers.

Publishing is a separate step; see hpc/report_site/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HPC_DIR = Path(__file__).resolve().parent
SITE_SOURCES = HPC_DIR / "report_site"
ATTACK = "pacman_attack"
KIND_ORDER = {"trained": 0, "untrained": 1, "baseline": 2}

SLOT = re.compile(r"<!-- slot:([\w-]+) -->")
SECTION = re.compile(r'<section data-slot="([\w-]+)">(.*?)</section>', re.DOTALL)
NARRATIVE_RUN = re.compile(r'<article data-run="([^"]+)">')
DATA_TAG = '<script src="report_data.js"></script>'


def _clean(value, digits: int = 4):
    if isinstance(value, (float, np.floating)):
        return (
            None
            if math.isnan(value) or math.isinf(value)
            else round(float(value), digits)
        )
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[list]:
    return [
        [_clean(v, digits) for v in row]
        for row in frame[columns].itertuples(index=False)
    ]


# --------------------------------------------------------------------------------------
# Sweep and selections
# --------------------------------------------------------------------------------------


def _feature_counts(run_root: Path) -> dict[str, int]:
    """Attack-sequence feature sets in the sweep and their column counts, fewest first."""
    counts = {}
    for result_path in (run_root / "results").glob("*.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result.get("config", {})
        if config.get("sequence_type") == ATTACK and result.get("n_features"):
            counts[config["feature_set"]] = int(result["n_features"])
    return dict(sorted(counts.items(), key=lambda item: item[1]))


def _selections(run_root: Path, manifest: dict) -> dict:
    """The manifest's rebuilt spaces, each with its configuration and what kind of space it is."""
    selections = {}
    for key, entry in manifest["selections"].items():
        config = json.loads(
            (run_root / entry["run_dir"] / "config.json").read_text(encoding="utf-8")
        )
        untrained = bool(entry.get("test_initialized_model"))
        selections[key] = {
            **entry,
            "kind": "untrained"
            if untrained
            else ("baseline" if config["embedder"] is None else "trained"),
            "config": {
                "sequence_type": config["sequence_type"],
                "feature_set": config["feature_set"],
                "embedder": config["embedder"],
                "latent_space": config["latent_space"],
            },
        }
    return selections


def _control(selections: dict) -> dict | None:
    for key, entry in selections.items():
        if entry["kind"] == "untrained":
            return {
                "key": key,
                **{
                    k: entry["config"][k]
                    for k in ("embedder", "feature_set", "latent_space")
                },
            }
    return None


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _web_copy(src: Path, dst: Path, max_width: int) -> None:
    from PIL import Image

    image = Image.open(src)
    if image.width > max_width:
        image = image.resize(
            (max_width, round(image.height * max_width / image.width)), Image.LANCZOS
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(dst.with_suffix(".jpg"), quality=86, optimize=True)


def _web_figures(report: Path, site: Path, selections: dict) -> dict:
    """Web-sized JPGs of every report figure, and an index of which spaces have which figures."""
    fig_dir = site / "fig"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)  # generated output only; stale figures must not linger
    for src in (report / "figures").glob("*.png"):
        _web_copy(src, fig_dir / src.name, 2000)

    clusters = {}
    for key_dir in sorted(p for p in (report / "clusters").iterdir() if p.is_dir()):
        ids = []
        for src in sorted(key_dir.glob("cluster_*.png")):
            _web_copy(src, fig_dir / "clusters" / key_dir.name / src.name, 1400)
            ids.append(int(src.stem.split("_")[1]))
        if key_dir.name in selections:
            clusters[key_dir.name] = ids

    # Trained spaces first, then the untrained control, then the no-encoder baseline.
    figures = report / "figures"
    ordered = sorted(selections, key=lambda k: KIND_ORDER[selections[k]["kind"]])
    return {
        "focus": [
            k for k in ordered if (figures / f"{k}_focus_behavlets.png").exists()
        ],
        "grids": [
            k
            for k in ordered
            if selections[k]["config"]["sequence_type"] == ATTACK
            and selections[k]["kind"] != "untrained"
            and (figures / f"{k}_validation_grid.png").exists()
        ],
        "clusters": clusters,
    }


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def _rq1(report: Path) -> dict:
    long = pd.read_csv(report / "rq1" / "validation_long.csv")
    long["set"] = long["validation_set"].str.replace("_value", "", regex=False)
    long["latent_space"] = long["latent_space"].fillna(0).astype(int)  # 0 = no encoder
    columns = [
        "sequence_type",
        "feature_set",
        "embedder",
        "latent_space",
        "set",
        "neigh_hit_adj",
        "neigh_hit",
        "NMI",
        "ARI",
    ]

    control_rows = []
    control_path = report / "rq1" / "untrained_control.csv"
    if control_path.exists():
        control = pd.read_csv(control_path)
        control["set"] = control["validation_set"].str.replace(
            "_value", "", regex=False
        )
        control_rows = _records(
            control, ["model", "set", "neigh_hit_adj", "neigh_hit", "NMI", "ARI"]
        )
    return {
        "columns": columns,
        "rows": _records(long, columns),
        "control": control_rows,
    }


def _rq2(report: Path, selections: dict) -> dict:
    spaces = {}
    for key, entry in selections.items():
        measures_path = report / "rq2" / f"{key}_participant_measures.csv"
        if not measures_path.exists():
            continue
        sequences = pd.read_csv(report / "latent" / f"{key}_sequences.csv")
        clusters = pd.read_csv(report / "latent" / f"{key}_clusters.csv")
        measures = pd.read_csv(measures_path)
        series = pd.read_csv(report / "rq2" / f"{key}_sequence_series.csv")
        correlations = pd.read_csv(report / "rq2" / f"{key}_correlations.csv")
        spaces[key] = {
            "parameters": entry.get("rq2_parameters", {}),
            "points": _records(sequences, ["x", "y", "cluster", "user_id"], 3),
            "clusters": _records(clusters, ["cluster", "size", "x", "y"], 3),
            "measure_columns": list(measures.columns),
            "measures": _records(measures, list(measures.columns)),
            "series": {
                str(int(user)): _records(
                    frame, ["x", "y", "cluster", "step_novelty", "pop_novelty"], 3
                )
                for user, frame in series.sort_values(["user_id", "step"]).groupby(
                    "user_id"
                )
            },
            "correlations": _records(
                correlations, ["a", "b", "r", "p", "n", "tested", "p_bonferroni"]
            ),
        }
    stats_path = report / "rq2" / "participant_stats.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    return {
        "stats_columns": list(stats.columns),
        "stats": _records(stats, list(stats.columns)),
        "spaces": spaces,
    }


# --------------------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------------------


def _narrative(run_id: str) -> tuple[dict[str, str], str | None]:
    """The narrative's slots, and the run it was written for."""
    text = (SITE_SOURCES / "narrative.html").read_text(encoding="utf-8")
    written_for = NARRATIVE_RUN.search(text)
    return {name: body.strip() for name, body in SECTION.findall(text)}, (
        written_for.group(1) if written_for else None
    )


def _stale_warning(written_for: str | None, run_id: str) -> str:
    if written_for == run_id:
        return ""
    return (
        '<p class="callout"><span><strong>Interpretation out of date.</strong> The text on this '
        f'page was written for run <span class="mono">{written_for or "unknown"}</span>; the '
        f'numbers come from run <span class="mono">{run_id}</span>. Re-read the findings, the '
        "section texts and the caveats against the new numbers before sharing.</span></p>"
    )


# --------------------------------------------------------------------------------------


def build_site(run_root: Path) -> Path:
    run_root = Path(run_root).resolve()
    report = run_root / "report"
    site = report / "site"
    site.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    run_id = run_root.name
    selections = _selections(run_root, manifest)

    data = {
        "run": run_id,
        "focus": [name.replace("_value", "") for name in manifest["focus_behavlets"]],
        "feature_sets": _feature_counts(run_root),
        "selections": selections,
        "control": _control(selections),
        "figures": _web_figures(report, site, selections),
        "rq1": _rq1(report),
        "rq2": _rq2(report, selections),
    }

    slots, written_for = _narrative(run_id)
    slots["stale-warning"] = _stale_warning(written_for, run_id)
    template = (SITE_SOURCES / "template.html").read_text(encoding="utf-8")
    missing = sorted(set(SLOT.findall(template)) - set(slots))
    if missing:
        raise SystemExit(
            f"narrative.html has no section for slot(s): {', '.join(missing)}"
        )
    unused = sorted(set(slots) - set(SLOT.findall(template)))
    if unused:
        print(
            f"Warning: narrative.html sections without a slot in the template: {', '.join(unused)}"
        )
    if template.count(DATA_TAG) != 1:
        raise SystemExit(f"template.html must contain exactly one {DATA_TAG}")

    page = SLOT.sub(lambda match: slots[match.group(1)], template)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(DATA_TAG, f"<script>window.REPORT_DATA = {payload};</script>")
    out = site / "report.html"
    out.write_text(page, encoding="utf-8")

    size = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    print(
        f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB page, {size / 1e6:.1f} MB with figures)"
    )
    if written_for != run_id:
        print(
            f"Warning: narrative.html was written for run {written_for}, not {run_id}; the page says so."
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run",
        required=True,
        help="A run directory whose report/ folder build_report_data.py filled.",
    )
    args = parser.parse_args()
    build_site(Path(args.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
