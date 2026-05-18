# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # TextAdaptive Scenario
#
# The `TextAdaptive` scenario uses an **epsilon-greedy selector** to intelligently choose
# which attack technique to try for each objective. Unlike static scenarios that run every
# selected technique against every objective, `TextAdaptive` adapts its strategy selection
# based on observed success rates — spending more attempts on techniques that work and
# exploring new ones with a configurable probability.
#
# ## How It Works
#
# For each objective (prompt), the selector:
#
# 1. **Explores** with probability `epsilon` — picks a technique uniformly at random.
# 2. **Exploits** otherwise — picks the technique with the highest observed success rate.
# 3. **Stops early** when a technique succeeds, avoiding wasted attempts.
# 4. Tries **up to** `max_attempts_per_objective` techniques before moving on.
#
# Unseen techniques start with an optimistic prior (100% success estimate), so the first
# few objectives effectively round-robin through every available technique before the
# selector converges on the best performers.
#
# ## Key Differences from Static Scenarios
#
# | Feature | Static Scenarios | TextAdaptive |
# |---------|-----------------|--------------|
# | Technique selection | Run all selected techniques | Selector picks per-objective |
# | Early stopping | No | Yes — stops on first success |
# | Learning | None | Updates success rates after each attempt |
# | Baseline | Prepended automatically | Forbidden — `prompt_sending` is a technique |
# | Efficiency | O(techniques × objectives) | O(max_attempts × objectives) |

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

from pyrit.registry import TargetRegistry
from pyrit.scenario import DatasetConfiguration
from pyrit.scenario.printer.console_printer import ConsoleScenarioResultPrinter
from pyrit.scenario.scenarios.adaptive import TextAdaptive, harm_category_context
from pyrit.setup import initialize_from_config_async

await initialize_from_config_async(config_path=Path("../../scanner/pyrit_conf.yaml"))  # type: ignore

objective_target = TargetRegistry.get_registry_singleton().get_instance_by_name("openai_chat")
printer = ConsoleScenarioResultPrinter()

# %% [markdown]
# ## Basic Usage
#
# The simplest way to run `TextAdaptive` uses all defaults: the selector explores with 20%
# probability, tries up to 3 techniques per objective, and uses the default dataset
# (AIRT harm categories).

# %%
scenario = TextAdaptive()

await scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
)
result = await scenario.run_async()  # type: ignore
await printer.print_summary_async(result)  # type: ignore

# %% [markdown]
# ## Customizing the Selector
#
# ### Epsilon (Exploration Rate)
#
# `epsilon` controls how often the selector explores vs. exploits:
# - `epsilon=0.0` — pure exploitation (always pick the best-known technique)
# - `epsilon=1.0` — pure exploration (random selection every time)
# - `epsilon=0.2` (default) — 20% random exploration, 80% exploitation

# %%
# More explorative selector — useful when you want broader technique coverage
explorative_scenario = TextAdaptive(epsilon=0.5)

await explorative_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=4),
)
explorative_result = await explorative_scenario.run_async()  # type: ignore
await printer.print_summary_async(explorative_result)  # type: ignore

# %% [markdown]
# ### Max Attempts Per Objective
#
# `max_attempts_per_objective` caps how many techniques the selector tries before giving
# up on an objective. Setting this higher gives more chances to succeed but costs more
# API calls.

# %%
persistent_scenario = TextAdaptive(max_attempts_per_objective=5)

await persistent_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_violence"], max_dataset_size=4),
)
persistent_result = await persistent_scenario.run_async()  # type: ignore
await printer.print_summary_async(persistent_result)  # type: ignore

# %% [markdown]
# ## Context-Aware Selection
#
# By default, the selector shares one global table across all objectives. This means
# a technique that works well on hate-speech objectives also gets boosted for
# violence objectives.
#
# To partition the selector by harm category (so each category learns independently),
# pass `harm_category_context` as the `context_extractor`:

# %%
contextual_scenario = TextAdaptive(
    context_extractor=harm_category_context,
    pool_threshold=2,
)

await contextual_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(
        dataset_names=["airt_hate", "airt_violence"],
        max_dataset_size=4,
    ),
)
contextual_result = await contextual_scenario.run_async()  # type: ignore
await printer.print_summary_async(contextual_result)  # type: ignore

# %% [markdown]
# The `pool_threshold` parameter controls how many local observations are needed before
# the per-category estimate overrides the pooled-global estimate. With
# `pool_threshold=2`, the selector uses the global average until it has seen at least 2
# results for a specific (category, technique) pair.

# %% [markdown]
# ## Strategy Selection
#
# `TextAdaptive` builds its strategy enum dynamically from the scenario-techniques
# catalog. You can restrict which techniques participate using the
# `scenario_strategies` parameter:

# %%
strategy_class = TextAdaptive.get_strategy_class()

# See all available strategies
print("Available strategies:")
for member in strategy_class:
    print(f"  {member.value}")

# %% [markdown]
# To limit the selector to only single-turn techniques:

# %%
single_turn_scenario = TextAdaptive()

await single_turn_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    scenario_strategies=[strategy_class("single_turn")],
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=4),
)
single_turn_result = await single_turn_scenario.run_async()  # type: ignore
await printer.print_summary_async(single_turn_result)  # type: ignore

# %% [markdown]
# ## Deterministic Runs
#
# For reproducibility, pass a `seed` to make the selector's random decisions deterministic:

# %%
deterministic_scenario = TextAdaptive(seed=42, epsilon=0.3)

await deterministic_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=2),
)
deterministic_result = await deterministic_scenario.run_async()  # type: ignore
await printer.print_summary_async(deterministic_result)  # type: ignore

# %% [markdown]
# ## Custom Scorer
#
# By default, `TextAdaptive` uses the standard composite scorer. You can override it
# with any `TrueFalseScorer`:

# %%
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskRefusalScorer, TrueFalseInverterScorer

refusal_scorer = SelfAskRefusalScorer(chat_target=OpenAIChatTarget())
inverted_scorer = TrueFalseInverterScorer(scorer=refusal_scorer)

custom_scorer_scenario = TextAdaptive(objective_scorer=inverted_scorer)

await custom_scorer_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=2),
)
custom_result = await custom_scorer_scenario.run_async()  # type: ignore
await printer.print_summary_async(custom_result)  # type: ignore

# %% [markdown]
# ## Notes
#
# - **No baseline**: `TextAdaptive` has `BASELINE_POLICY = Forbidden`. The `prompt_sending`
#   technique participates as one of the selector's techniques, so a separate baseline is redundant.
# - **Resumability**: Each atomic attack is keyed by `adaptive_{dataset}_{objective_id}`, so
#   re-running a scenario picks up where it left off.
# - **Shared selector**: All objectives in a run share the same `AdaptiveTechniqueSelector`
#   instance, so learning from one objective immediately benefits the next.
