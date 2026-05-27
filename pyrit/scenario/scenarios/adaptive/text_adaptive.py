# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``TextAdaptive`` — text adaptive scenario.

Picks attack techniques per-objective using an epsilon-greedy selector
informed by observed success rates. Runs up to ``max_attempts_per_objective``
techniques per objective and stops early on success. ``prompt_sending`` is
excluded from the adaptive technique pool and runs as the baseline comparison
instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from pyrit.common import apply_defaults
from pyrit.common.parameter import Parameter
from pyrit.registry.tag_query import TagQuery
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.scenarios.adaptive.adaptive_scenario import AdaptiveScenario

if TYPE_CHECKING:
    from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
    from pyrit.scenario.scenarios.adaptive.selectors import TechniqueSelector
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)

# Techniques excluded from the adaptive technique pool. These run as the
# baseline comparison rather than as adversarial moves the selector chooses.
_EXCLUDED_TECHNIQUES = frozenset({"prompt_sending"})


def _build_text_adaptive_strategy() -> type[ScenarioStrategy]:
    """
    Build the strategy enum from the core scenario-techniques catalog,
    excluding techniques that run as baseline.

    Returns:
        type[ScenarioStrategy]: The dynamically-built strategy enum class.
    """
    from pyrit.registry.object_registries.attack_technique_registry import (
        AttackTechniqueRegistry,
    )
    from pyrit.scenario.core.scenario_techniques import SCENARIO_TECHNIQUES

    filtered_specs = [spec for spec in SCENARIO_TECHNIQUES if spec.name not in _EXCLUDED_TECHNIQUES]

    return AttackTechniqueRegistry.build_strategy_class_from_specs(  # type: ignore[return-value, ty:invalid-return-type]
        class_name="TextAdaptiveStrategy",
        specs=filtered_specs,
        aggregate_tags={
            "default": TagQuery.any_of("default"),
            "single_turn": TagQuery.any_of("single_turn"),
            "multi_turn": TagQuery.any_of("multi_turn"),
        },
    )


class TextAdaptive(AdaptiveScenario):
    """
    Adaptive text-attack scenario.

    Selects techniques per-objective via an epsilon-greedy selector over the
    set of selected strategies. ``prompt_sending`` runs as the baseline
    comparison and is excluded from the adaptive technique pool.
    """

    _cached_strategy_class: ClassVar[type[ScenarioStrategy] | None] = None

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        """Return the strategy enum for this scenario, building it once on first access."""
        if cls._cached_strategy_class is None:
            cls._cached_strategy_class = _build_text_adaptive_strategy()
        return cls._cached_strategy_class

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        """Return the default strategy aggregate (resolves to every ``default``-tagged technique)."""
        strategy_class = cls.get_strategy_class()
        return strategy_class("default")

    @classmethod
    def required_datasets(cls) -> list[str]:
        """Return the dataset names this scenario expects when no override is provided."""
        return [
            "airt_hate",
            "airt_fairness",
            "airt_violence",
            "airt_sexual",
            "airt_harassment",
            "airt_misinformation",
            "airt_leakage",
        ]

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        """Return the default :class:`DatasetConfiguration` (required datasets, capped at 4 per dataset)."""
        return DatasetConfiguration(dataset_names=cls.required_datasets(), max_dataset_size=4)

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Declare custom parameters this scenario accepts from the CLI / config file.

        Returns:
            list[Parameter]: Parameters configurable per-run.
        """
        return [
            Parameter(
                name="max_attempts_per_objective",
                description="Max techniques tried per objective.",
                param_type=int,
                default=3,
            ),
        ]

    @apply_defaults
    def __init__(
        self,
        *,
        objective_scorer: TrueFalseScorer | None = None,
        selector: TechniqueSelector | None = None,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Args:
            objective_scorer (TrueFalseScorer | None): Scorer used to judge each
                response. Defaults to the composite scorer from the base class.
            selector (TechniqueSelector | None): Pre-built selector. When ``None``
                (default) an :class:`EpsilonGreedyTechniqueSelector` is created
                with default settings. Pass a custom instance to tune
                ``epsilon`` or ``random_seed``.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult`` to resume.
        """
        super().__init__(
            objective_scorer=objective_scorer,
            selector=selector,
            scenario_result_id=scenario_result_id,
        )
