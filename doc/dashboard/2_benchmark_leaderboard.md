# Benchmark Leaderboard

Attack success rate (ASR) for adversarial techniques and models, built from `AdversarialBenchmark`
scenario runs. See [Benchmark Scenarios](../scanner/benchmark.ipynb) for how to run the scenario
yourself and `build_scripts/export_adversarial_benchmark_result.py` for how a run's results land
in the committed store this page reads.

## Technique / Adversarial Model Leaderboard

Ranked by success rate — the share of objectives where the target's response was scored as
achieving the objective — descending. `N` is the number of objectives attempted for that
(technique, adversarial model, objective target, objective scorer, dataset) combination; treat
rows with a small `N` as directional, not statistically robust.

```{include} _generated/benchmark_leaderboard.html
```

## Note on scope

This table is a snapshot upserted by `--update-benchmark-store`, keyed on
`(technique, adversarial_model, objective_target, objective_scorer, dataset)` — a fresh run of
the same combination replaces its row rather than appending, so this stays a single
best-known-result table rather than an unbounded history. Small `N` values (as in the rows
above) reflect small demo-scale runs (`--max-dataset-size`), not a statistically robust
evaluation; treat them as a preview of the mechanism rather than a final verdict on any
technique or model. Widening this into a larger, regularly-refreshed sweep — and adding a
native objective-target comparison alongside the current adversarial-model comparison — is
future work.

This leaderboard is a static snapshot, not a live view: it's rendered from the committed JSONL
store by `python -m build_scripts.generate_dashboard_html` (see
[Refreshing the data](0_dashboard.md#refreshing-the-data)), not computed when this page loads.
