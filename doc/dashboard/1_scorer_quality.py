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
# # Scorer Quality
#
# Leaderboards for the scorer evaluation metrics PyRIT already tracks under
# `pyrit/datasets/scorer_evals/`. See [Scorer Metrics](../code/scoring/4_scorer_metrics.ipynb) for
# what each metric means and how these numbers are produced.

# %% [markdown]
# ## Objective Scorer Leaderboard
#
# Objective scorers answer a true/false question (e.g. "was the objective achieved?"). Ranked by
# F1 score, the harmonic mean of precision and recall.

# %%
import pandas as pd

from pyrit.score import get_all_objective_metrics
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY, silent=True)  # type: ignore

objective_metrics = get_all_objective_metrics()
objective_metrics.sort(key=lambda entry: entry.metrics.f1_score, reverse=True)

objective_rows = [
    {
        "Name": entry.scorer_identifier.unique_name,
        "Accuracy": entry.metrics.accuracy,
        "F1 Score": entry.metrics.f1_score,
        "Precision": entry.metrics.precision,
        "Recall": entry.metrics.recall,
        "Samples": entry.metrics.num_responses,
    }
    for entry in objective_metrics
]

objective_df = pd.DataFrame(objective_rows)
pd.set_option("display.max_rows", None)
print(objective_df.to_string(index=False))

# %% [markdown]
# ## Harm Scorer Leaderboard
#
# Harm scorers produce a severity score (0.0-1.0). Ranked by `krippendorff_alpha_combined` —
# agreement between the model's scores and human raters, ranging from -1.0 (systematic
# disagreement) to 1.0 (perfect agreement) — across every harm category PyRIT currently has
# metrics for. Alpha isn't comparable *across* categories (each has its own human-labeled
# dataset), so treat this as one leaderboard per category, stacked into a single table for
# convenience.

# %%
from pyrit.common.path import SCORER_EVALS_HARM_PATH
from pyrit.score import get_all_harm_metrics

# Harm categories are discovered from the files present on disk rather than a hardcoded list,
# so a newly added category shows up here without a code change.
harm_categories = sorted(
    path.name.removesuffix("_metrics.jsonl") for path in SCORER_EVALS_HARM_PATH.glob("*_metrics.jsonl")
)

harm_metrics = [
    (harm_category, entry)
    for harm_category in harm_categories
    for entry in get_all_harm_metrics(harm_category=harm_category)
]
harm_metrics.sort(key=lambda item: item[1].metrics.krippendorff_alpha_combined, reverse=True)

harm_rows = [
    {
        "Name": entry.scorer_identifier.unique_name,
        "Harm Category": harm_category,
        "MAE": entry.metrics.mean_absolute_error,
        "Alpha Combined": entry.metrics.krippendorff_alpha_combined,
        "Alpha Humans": entry.metrics.krippendorff_alpha_humans,
        "Alpha Model": entry.metrics.krippendorff_alpha_model,
        "Samples": entry.metrics.num_responses,
    }
    for harm_category, entry in harm_metrics
]

harm_df = pd.DataFrame(harm_rows)
print(harm_df.to_string(index=False))

# %% [markdown]
# ## Note on scope
#
# `get_all_objective_metrics()` reads `objective/objective_achieved_metrics.jsonl` only, matching
# how it's documented and used elsewhere in PyRIT. A separate `refusal_scorer/refusal_metrics.jsonl`
# registry evaluates refusal scorers against its own human-labeled dataset, using the same
# `ObjectiveScorerMetrics` shape. It isn't merged into the leaderboard above because it measures a
# different task (refusal detection, not objective achievement) against a different ground truth
# set, and mixing the two would make the F1 ranking misleading. A follow-up could add it as its own
# leaderboard.
