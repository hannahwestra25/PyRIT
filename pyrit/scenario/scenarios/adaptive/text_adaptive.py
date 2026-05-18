# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
TextAdaptive scenario — picks attack techniques per-objective using an
epsilon-greedy selector informed by observed per-run success rates.

Unlike static scenarios (which run every selected technique against every
objective), TextAdaptive runs **up to** ``max_attempts_per_objective``
techniques per objective and stops early when one succeeds. Which technique
to try next is decided by an ``AdaptiveTechniqueSelector`` whose estimates are
updated after every attempt.

The set of available techniques comes from the selected scenario strategies, so
``--strategies single_turn`` restricts the selector to single-turn techniques,
etc. The default selector uses a single global context; pass a different
``context_extractor`` (e.g., ``harm_category_context``) to partition estimates
per category.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pyrit.common import apply_defaults
from pyrit.executor.attack import AttackScoringConfig
from pyrit.registry.tag_query import TagQuery
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselinePolicy, Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_CONTEXT_LABEL,
    AdaptiveDispatchAttack,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    AdaptiveTechniqueSelector,
    ContextExtractor,
    global_context,
)

if TYPE_CHECKING:
    from pyrit.executor.attack.core.attack_strategy import AttackStrategy
    from pyrit.models import AttackResult, SeedAttackGroup
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


def _build_text_adaptive_strategy() -> type[ScenarioStrategy]:
    """Build the strategy enum from the core scenario-techniques catalog."""
    from pyrit.registry.object_registries.attack_technique_registry import (
        AttackTechniqueRegistry,
    )
    from pyrit.scenario.core.scenario_techniques import SCENARIO_TECHNIQUES

    return AttackTechniqueRegistry.build_strategy_class_from_specs(  # type: ignore[return-value, ty:invalid-return-type]
        class_name="TextAdaptiveStrategy",
        specs=SCENARIO_TECHNIQUES,
        aggregate_tags={
            "default": TagQuery.any_of("default"),
            "single_turn": TagQuery.any_of("single_turn"),
            "multi_turn": TagQuery.any_of("multi_turn"),
        },
    )


class TextAdaptive(Scenario):
    """
    Adaptive text-attack scenario that selects techniques per-objective using
    an epsilon-greedy selector over the set of selected strategies.

    The selector:
        - Picks a technique uniformly at random with probability ``epsilon``.
        - Otherwise exploits the highest observed success rate. Unseen techniques
          have an optimistic prior so the first few objectives effectively
          round-robin through every available technique.
        - Pools across contexts when a context has fewer than
          ``pool_threshold`` observations for a technique.

    A baseline ``PromptSendingAttack`` is **not** prepended — every objective
    runs through the dispatcher, and ``prompt_sending`` participates as one of
    the selector's techniques.
    """

    VERSION: int = 1
    BASELINE_POLICY: ClassVar[BaselinePolicy] = BaselinePolicy.Forbidden
    _cached_strategy_class: ClassVar[type[ScenarioStrategy] | None] = None

    # ------------------------------------------------------------------ #
    # Required class-method overrides                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        if cls._cached_strategy_class is None:
            cls._cached_strategy_class = _build_text_adaptive_strategy()
        return cls._cached_strategy_class

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        strategy_class = cls.get_strategy_class()
        return strategy_class("default")

    @classmethod
    def required_datasets(cls) -> list[str]:
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
        return DatasetConfiguration(dataset_names=cls.required_datasets(), max_dataset_size=4)

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #

    @apply_defaults
    def __init__(
        self,
        *,
        objective_scorer: TrueFalseScorer | None = None,
        epsilon: float = 0.2,
        pool_threshold: int = 3,
        max_attempts_per_objective: int = 3,
        seed: int | None = None,
        context_extractor: ContextExtractor = global_context,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Args:
            objective_scorer (TrueFalseScorer | None): Scorer used to judge each
                response. Defaults to the composite scorer built from the base class.
            epsilon (float): Exploration probability for the selector. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before the
                local estimate overrides the pooled-global estimate. Set to 1 to
                disable pooling. Defaults to 3.
            max_attempts_per_objective (int): Maximum techniques tried per
                objective before giving up. Defaults to 3.
            seed (int | None): RNG seed for deterministic selection decisions.
                Defaults to ``None`` (non-deterministic).
            context_extractor (ContextExtractor): Function mapping a
                ``SeedAttackGroup`` to a context key. Defaults to
                ``global_context`` (one shared selection table). Use
                ``harm_category_context`` to partition estimates by harm category.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult``
                to resume.
        """
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()

        self._epsilon = epsilon
        self._pool_threshold = pool_threshold
        self._max_attempts_per_objective = max_attempts_per_objective
        self._seed = seed
        self._context_extractor = context_extractor

        super().__init__(
            version=self.VERSION,
            strategy_class=self.get_strategy_class(),
            objective_scorer=objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    # ------------------------------------------------------------------ #
    # Override atomic-attack construction                                #
    # ------------------------------------------------------------------ #

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Build one ``AtomicAttack`` per objective, all sharing a single
        ``AdaptiveDispatchAttack`` (and therefore a single
        ``AdaptiveTechniqueSelector``).

        Each per-objective ``AtomicAttack`` consults and updates the same
        selector via the same dispatcher instance, so learning from one
        objective immediately benefits the next.
        """
        if self._objective_target is None:
            raise ValueError("objective_target must be set before creating attacks")

        selected_techniques = sorted({s.value for s in self._scenario_strategies})
        factories = self._get_attack_technique_factories()

        # Build each technique's inner attack once and reuse across all objectives.
        # Skip factories that require a seed_technique (e.g. crescendo_simulated)
        # since the dispatcher cannot merge technique seeds into the objective's
        # seed group at dispatch time.
        scoring_config = AttackScoringConfig(objective_scorer=cast("TrueFalseScorer", self._objective_scorer))
        techniques: dict[str, AttackStrategy[Any, AttackResult]] = {}
        for technique_name in selected_techniques:
            factory = factories.get(technique_name)
            if factory is None:
                logger.warning(f"No factory for technique '{technique_name}', skipping.")
                continue
            technique = factory.create(
                objective_target=self._objective_target,
                attack_scoring_config=scoring_config,
            )
            if technique.seed_technique is not None:
                logger.debug(
                    "Skipping technique '%s': requires seed_technique which adaptive dispatch cannot handle.",
                    technique_name,
                )
                continue
            techniques[technique_name] = technique.attack

        if not techniques:
            raise ValueError(
                "TextAdaptive: no usable techniques after resolving strategies. Check the --strategies selection."
            )

        selector = AdaptiveTechniqueSelector(
            epsilon=self._epsilon,
            pool_threshold=self._pool_threshold,
            rng=random.Random(self._seed),
        )
        dispatcher = AdaptiveDispatchAttack(
            objective_target=self._objective_target,
            techniques=techniques,
            selector=selector,
            max_attempts_per_objective=self._max_attempts_per_objective,
        )

        seed_groups_by_dataset = self._dataset_config.get_seed_attack_groups()
        atomic_attacks: list[AtomicAttack] = []
        for dataset_name, seed_groups in seed_groups_by_dataset.items():
            for seed_group in seed_groups:
                atomic_attacks.append(
                    self._build_atomic_for_seed_group(
                        dataset_name=dataset_name,
                        seed_group=seed_group,
                        dispatcher=dispatcher,
                    )
                )

        return atomic_attacks

    def _build_atomic_for_seed_group(
        self,
        *,
        dataset_name: str,
        seed_group: SeedAttackGroup,
        dispatcher: AdaptiveDispatchAttack,
    ) -> AtomicAttack:
        from pyrit.scenario.core.atomic_attack import AtomicAttack
        from pyrit.scenario.core.attack_technique import AttackTechnique

        adaptive_context = self._context_extractor(seed_group)
        # Use the objective's id when available so resume keys are stable across
        # runs that re-fetch the same seed groups; fall back to a random uuid.
        objective_id = seed_group.objective.id if seed_group.objective.id else uuid.uuid4()
        atomic_attack_name = f"adaptive_{dataset_name}_{objective_id}"

        memory_labels = {
            **self._memory_labels,
            ADAPTIVE_CONTEXT_LABEL: adaptive_context,
        }
        return AtomicAttack(
            atomic_attack_name=atomic_attack_name,
            attack_technique=AttackTechnique(attack=dispatcher),
            seed_groups=[seed_group],
            objective_scorer=cast("TrueFalseScorer", self._objective_scorer),
            memory_labels=memory_labels,
            display_group=dataset_name,
        )
