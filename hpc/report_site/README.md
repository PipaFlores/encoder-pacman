# Results report page

The *Pac-Man Latent Space Results* page (RQ1 latent structure, RQ2 participant dynamics) is
built from a finished sweep in three steps. The page is published as a private claude.ai
artifact: <https://claude.ai/artifact/63z6qJTnMA39pP9eteHmAM>.

| File | What it is | Changes when |
|---|---|---|
| `hpc/build_report_data.py` | Rebuilds the chosen spaces from their sweep checkpoints (never retrains) and writes numbers and figures to `<run>/report/` | which configurations to show (`SELECTIONS`, `FOCUS_BEHAVLETS`) |
| `hpc/build_report_site.py` | Turns `<run>/report/` into `<run>/report/site/report.html` plus `site/fig/` | rarely |
| `template.html` | Layout and interaction. Models, latent sizes, feature sets and spaces are read from the data | the page's design or a new kind of chart |
| `narrative.html` | The interpretation: findings, section texts, methods, caveats | **every rebuild from new results** |

## Updating the page

1. **Run and collect the sweep.** After syncing it back from the cluster:
   `python hpc/run_experiment.py --config hpc/experiments/<sweep>.yaml --run-id <run-id> --collect`
2. **Rebuild the data and the page** (local, CPU, a few minutes once Behavlets are cached):
   `python hpc/build_report_data.py --run hpc/runs/<sweep>/<run-id> --site`
   It prints how closely each rebuilt space matches the sweep (`reproduction:`).
3. **Re-read `narrative.html` against the new numbers.** Rewrite whatever no longer holds, then
   set `<article data-run="...">` to the new run id. Until then the page shows an
   *Interpretation out of date* warning.
4. **Rebuild the page only**, which takes seconds:
   `python hpc/build_report_site.py --run hpc/runs/<sweep>/<run-id>`
5. **Publish.** Ask Claude to republish `<run>/report/site/report.html`, with `fig/` as its
   files, to the artifact URL above. Republishing to that URL keeps the link and its version
   history. The code font (`fonts/Hack-Regular.ttf`) is already part of the artifact.

Run everything from the repository root with the `pacman_encoder` environment active and
`PYTHONPATH=.`.

## What adapts on its own, and what does not

- **Adapts:** new encoders, latent sizes and feature sets in the sweep (comparison table, line
  charts, controls), the rebuilt spaces listed in `manifest.json` (view buttons, cluster maps,
  participant explorer, provenance table), and the run id.
- **Needs an edit:** which configurations are rebuilt (`SELECTIONS` in
  `build_report_data.py`), the focus Behavlets (`FOCUS_BEHAVLETS`), the participant-measure
  lists in `template.html`, and every sentence in `narrative.html`.
- **Several seeds per configuration:** `find_run()` expects exactly one run per selection, so add
  `seed` to the matching selection's criteria first.
- **New data (participants, sessions):** the checkpoints and the sweep's scores still describe
  the old data. Re-run the sweep, or at least the selected configurations, before step 2.
