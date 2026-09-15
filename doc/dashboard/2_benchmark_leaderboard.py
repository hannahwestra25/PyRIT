# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---
# %% [markdown]
# # Benchmark Leaderboard
#
# Attack success rate (ASR) for adversarial techniques and models, built from `AdversarialBenchmark`
# scenario runs. See [Benchmark Scenarios](../scanner/benchmark.ipynb) for how to run the scenario
# yourself and `build_scripts/export_adversarial_benchmark_result.py` for how a run's results land
# in the committed store this page reads.

# %% [markdown]
# ## Technique / Adversarial Model Leaderboard
#
# Ranked by success rate — the share of objectives where the target's response was scored as
# achieving the objective — descending. `N` is the number of objectives attempted for that
# (technique, adversarial model, objective target, objective scorer, dataset) combination; treat
# rows with a small `N` as directional, not statistically robust.

# %%
import pandas as pd

from pyrit.common.path import BENCHMARK_RESULTS_PATH

_STORE_PATH = BENCHMARK_RESULTS_PATH / "adversarial_benchmark_metrics.jsonl"

_COLUMN_LABELS = {
    "technique": "Technique",
    "adversarial_model": "Adversarial Model",
    "objective_target": "Objective Target",
    "objective_scorer": "Objective Scorer",
    "dataset": "Dataset",
    "total": "N",
    "success": "Success",
    "failure": "Failure",
    "error": "Error",
    "undetermined": "Undetermined",
    "retry_records": "Retries",
    "success_rate": "Success Rate",
}

if _STORE_PATH.exists():
    benchmark_df = pd.read_json(_STORE_PATH, lines=True)
    benchmark_df = benchmark_df.sort_values("success_rate", ascending=False)
    display_df = benchmark_df.rename(columns=_COLUMN_LABELS)[list(_COLUMN_LABELS.values())]
    pd.set_option("display.max_rows", None)
    print(display_df.to_string(index=False))
else:
    print(f"No benchmark data yet at {_STORE_PATH}.")

# %% [markdown]
# ## Note on scope
#
# This table is a snapshot upserted by `--update-benchmark-store`, keyed on
# `(technique, adversarial_model, objective_target, objective_scorer, dataset)` — a fresh run of
# the same combination replaces its row rather than appending, so this stays a single
# best-known-result table rather than an unbounded history. Small `N` values (as in the rows
# above) reflect small demo-scale runs (`--max-dataset-size`), not a statistically robust
# evaluation; treat them as a preview of the mechanism rather than a final verdict on any
# technique or model. Widening this into a larger, regularly-refreshed sweep — and adding a
# native objective-target comparison alongside the current adversarial-model comparison — is
# future work.
