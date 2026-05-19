# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""``AdaptiveScenario`` — modality-agnostic base for scenarios that pick attack
techniques per-objective using an ``AdaptiveTechniqueSelector``.

Owns selector wiring, dispatcher construction, per-objective atomic-attack
emission, and resume rehydration. Concrete subclasses (``TextAdaptive``,
future ``ImageAdaptive`` / ``AudioAdaptive``) only declare strategy class,
default datasets, version, and atomic-attack prefix.

Baseline policy is ``Forbidden``: ``prompt_sending`` participates as one of
the selector's techniques rather than being prepended.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from pyrit.executor.attack import AttackScoringConfig
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.scenario import BaselinePolicy, Scenario
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
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


class AdaptiveScenario(Scenario):
    """Abstract base for adaptive (epsilon-greedy) scenarios.

    Subclasses must implement the standard ``Scenario`` class-method overrides
    and declare ``VERSION`` and ``_atomic_attack_prefix``. Selector wiring,
    dispatcher construction, per-objective atomic-attack emission, and resume
    rehydration are handled here.
    """

    BASELINE_POLICY: ClassVar[BaselinePolicy] = BaselinePolicy.Forbidden

    #: Subclasses must declare a scenario version for memory bookkeeping.
    VERSION: ClassVar[int]

    #: Prefix for per-objective atomic-attack names (e.g. ``"adaptive_text"``).
    _atomic_attack_prefix: ClassVar[str] = "adaptive"

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
                response. Defaults to the composite scorer from the base class.
            epsilon (float): Exploration probability for the selector. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before
                the local estimate overrides the pooled rate. Set to 1 to disable
                pooling. Defaults to 3.
            max_attempts_per_objective (int): Max techniques per objective. Defaults to 3.
            seed (int | None): RNG seed for deterministic selection. Defaults to ``None``.
            context_extractor (ContextExtractor): Maps a ``SeedAttackGroup`` to a
                context key. Defaults to ``global_context``.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult`` to resume.
        """
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._objective_scorer: TrueFalseScorer = objective_scorer

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

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """Build one ``AtomicAttack`` per objective, all sharing a single
        ``AdaptiveDispatchAttack`` (and therefore a single selector).
        """
        if self._objective_target is None:
            raise ValueError("objective_target must be set before creating attacks")

        techniques = self._build_techniques_dict(objective_target=self._objective_target)

        selector = AdaptiveTechniqueSelector(
            epsilon=self._epsilon,
            pool_threshold=self._pool_threshold,
            rng=random.Random(self._seed),
        )
        # On resume, replay prior attempt outcomes from persisted metadata.
        self._rehydrate_selector_from_memory(selector=selector, known_techniques=set(techniques))

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

    def _build_techniques_dict(
        self,
        *,
        objective_target: PromptTarget,
    ) -> dict[str, AttackStrategy[Any, AttackResult]]:
        """Resolve selected strategies into a ``{name: inner_attack}`` map.

        Skips factories not registered for the current modality, and factories
        whose technique requires a ``seed_technique`` (e.g. ``crescendo_simulated``)
        — the dispatcher has no hook to merge technique seeds into per-objective
        seed groups.

        Raises:
            ValueError: If no techniques remain after filtering. Includes the
                requested techniques and skip reasons.
        """
        selected_techniques = sorted({s.value for s in self._scenario_strategies})
        factories = self._get_attack_technique_factories()
        scoring_config = AttackScoringConfig(objective_scorer=self._objective_scorer)

        techniques: dict[str, AttackStrategy[Any, AttackResult]] = {}
        skipped_seed_technique: list[str] = []
        skipped_no_factory: list[str] = []
        for technique_name in selected_techniques:
            factory = factories.get(technique_name)
            if factory is None:
                skipped_no_factory.append(technique_name)
                logger.warning(f"No factory for technique '{technique_name}', skipping.")
                continue
            technique = factory.create(
                objective_target=objective_target,
                attack_scoring_config=scoring_config,
            )
            if technique.seed_technique is not None:
                skipped_seed_technique.append(technique_name)
                logger.warning(
                    "Skipping technique '%s': it requires a seed_technique which the adaptive "
                    "dispatcher cannot merge into per-objective seed groups. Use a static "
                    "scenario (e.g. RapidResponse) to run this technique.",
                    technique_name,
                )
                continue
            techniques[technique_name] = technique.attack

        if not techniques:
            details: list[str] = []
            if skipped_seed_technique:
                details.append(f"skipped (require seed_technique): {sorted(skipped_seed_technique)}")
            if skipped_no_factory:
                details.append(f"skipped (no factory registered): {sorted(skipped_no_factory)}")
            suffix = f" ({'; '.join(details)})" if details else ""
            raise ValueError(
                f"{type(self).__name__}: no usable techniques after resolving strategies. "
                f"Check the --strategies selection.{suffix}"
            )

        return techniques

    def _build_atomic_for_seed_group(
        self,
        *,
        dataset_name: str,
        seed_group: SeedAttackGroup,
        dispatcher: AdaptiveDispatchAttack,
    ) -> AtomicAttack:
        adaptive_context = self._context_extractor(seed_group)
        # Prefer the objective's id when available so resume keys stay stable
        # across re-fetches of the same seed groups.
        objective_id = seed_group.objective.id if seed_group.objective.id else uuid.uuid4()
        atomic_attack_name = f"{self._atomic_attack_prefix}_{dataset_name}_{objective_id}"

        memory_labels = {
            **self._memory_labels,
            ADAPTIVE_CONTEXT_LABEL: adaptive_context,
        }
        return AtomicAttack(
            atomic_attack_name=atomic_attack_name,
            attack_technique=AttackTechnique(attack=dispatcher),
            seed_groups=[seed_group],
            objective_scorer=self._objective_scorer,
            memory_labels=memory_labels,
            display_group=dataset_name,
        )

    def _rehydrate_selector_from_memory(
        self,
        *,
        selector: AdaptiveTechniqueSelector,
        known_techniques: set[str],
    ) -> None:
        """Replay persisted dispatch trails into ``selector`` so resume
        preserves learned state.

        Iterates every persisted ``AttackResult`` on the resumed
        ``ScenarioResult`` and calls ``record_outcome`` once per attempt in
        each ``metadata["adaptive_attempts"]`` trail.

        Args:
            selector (AdaptiveTechniqueSelector): A freshly built selector to populate.
            known_techniques (set[str]): Techniques available in the current run.
                Trails referencing unknown techniques (e.g. after a strategies
                change) are skipped so replay can't poison the table.
        """
        if not self._scenario_result_id:
            return

        try:
            scenario_results = self._memory.get_scenario_results(scenario_result_ids=[self._scenario_result_id])
        except Exception as exc:
            logger.warning(f"AdaptiveScenario: failed to load prior scenario result for rehydration: {exc}")
            return

        if not scenario_results:
            return

        replayed = 0
        for results_list in scenario_results[0].attack_results.values():
            for result in results_list:
                trail = result.metadata.get("adaptive_attempts") if result.metadata else None
                context = result.metadata.get("adaptive_context") if result.metadata else None
                if not trail or not context:
                    continue
                for step in trail:
                    technique = step.get("technique")
                    outcome = step.get("outcome")
                    if not technique or technique not in known_techniques:
                        continue
                    selector.record_outcome(
                        context=context,
                        technique=technique,
                        success=outcome == "success",
                    )
                    replayed += 1

        if replayed:
            logger.info(f"AdaptiveScenario: rehydrated selector with {replayed} prior attempt(s).")
