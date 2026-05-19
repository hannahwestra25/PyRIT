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
# # Adaptive Scenarios
#
# An **adaptive scenario** doesn't run every attack technique against every objective.
# Instead, it picks which technique to try next per-objective, learns from what worked,
# and stops as soon as one technique succeeds. This concentrates spend on techniques
# that actually work on your target.
#
# ## How it works (high level)
#
# For each objective, the scenario tries up to `max_attempts_per_objective` techniques:
#
# - With probability `epsilon`, it **explores** — picks a random technique.
# - Otherwise it **exploits** — picks the technique with the highest observed success
#   rate so far.
# - It records the outcome and stops early on success.
#
# Unseen techniques are tried first, so the first few objectives effectively round-robin
# through every technique before the scenario settles on the best performers.
#
# ## Adaptive vs. static scenarios
#
# | Feature             | Static scenarios                  | Adaptive scenarios                 |
# |---------------------|-----------------------------------|------------------------------------|
# | Technique selection | Run every selected technique      | Pick per-objective from outcomes   |
# | Early stopping      | No                                | Yes — stops on first success       |
# | Cost                | O(techniques × objectives)        | O(max_attempts × objectives)       |
#
# `AdaptiveScenario` is the modality-agnostic base class.
# [`TextAdaptive`](../../../pyrit/scenario/scenarios/adaptive/text_adaptive.py) is the
# text subclass used in the examples below.

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
# ## Basic usage
#
# Defaults: `epsilon=0.2`, `max_attempts_per_objective=3`, the subclass's default datasets.

# %%
scenario = TextAdaptive()

await scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
)
result = await scenario.run_async()  # type: ignore
await printer.print_summary_async(result)  # type: ignore

# %% [markdown]
# ## Tuning exploration (`epsilon`)
#
# - `epsilon=0.0` — pure exploitation (always pick the best-known technique).
# - `epsilon=1.0` — pure exploration (random every time).
# - `epsilon=0.2` (default) — 20% exploration.

# %%
explorative_scenario = TextAdaptive(epsilon=0.5)

await explorative_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=4),
)
explorative_result = await explorative_scenario.run_async()  # type: ignore
await printer.print_summary_async(explorative_result)  # type: ignore

# %% [markdown]
# ## Attempts per objective
#
# `max_attempts_per_objective` caps how many techniques are tried per objective before
# moving on. Higher = more chances to succeed, more API calls.

# %%
persistent_scenario = TextAdaptive(max_attempts_per_objective=5)

await persistent_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_violence"], max_dataset_size=4),
)
persistent_result = await persistent_scenario.run_async()  # type: ignore
await printer.print_summary_async(persistent_result)  # type: ignore

# %% [markdown]
# ## Learning per harm category
#
# By default, the scenario keeps one global success-rate table — what works on hate
# objectives boosts the same technique on violence objectives. Pass `harm_category_context`
# to learn each category independently:

# %%
contextual_scenario = TextAdaptive(context_extractor=harm_category_context)

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
# ## Restricting which techniques participate
#
# Use `scenario_strategies` to limit which techniques the scenario can pick from.

# %%
strategy_class = TextAdaptive.get_strategy_class()

single_turn_scenario = TextAdaptive()

await single_turn_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    scenario_strategies=[strategy_class("single_turn")],
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=4),
)
single_turn_result = await single_turn_scenario.run_async()  # type: ignore
await printer.print_summary_async(single_turn_result)  # type: ignore

# %% [markdown]
# ## Reproducible runs
#
# Pass `seed` to make every selection decision deterministic.

# %%
deterministic_scenario = TextAdaptive(seed=42, epsilon=0.3)

await deterministic_scenario.initialize_async(  # type: ignore
    objective_target=objective_target,
    dataset_config=DatasetConfiguration(dataset_names=["airt_hate"], max_dataset_size=2),
)
deterministic_result = await deterministic_scenario.run_async()  # type: ignore
await printer.print_summary_async(deterministic_result)  # type: ignore

# %% [markdown]
# ## Resuming a run
#
# Adaptive scenarios are resumable — pass `scenario_result_id=...` to the `TextAdaptive`
# constructor and the run picks up where it left off, with prior outcomes replayed into
# the selector.
#
# ```python
# resumed_scenario = TextAdaptive(scenario_result_id="<existing-scenario-result-id>")
# await resumed_scenario.initialize_async(objective_target=objective_target)
# resumed_result = await resumed_scenario.run_async()
# ```
