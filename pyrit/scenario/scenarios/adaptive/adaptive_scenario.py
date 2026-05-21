# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveScenario`` — modality-agnostic base for scenarios that pick attack
techniques per-objective using a ``TechniqueSelector``.

Owns selector wiring, dispatcher construction, per-dataset atomic-attack
emission, and resume rehydration. Concrete subclasses (``TextAdaptive``,
future ``ImageAdaptive`` / ``AudioAdaptive``) only declare strategy class,
default datasets, version, and atomic-attack prefix.

Baseline policy is ``Enabled``: prompt_sending runs as a separate baseline
comparison and is excluded from the adaptive technique pool.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from pyrit.executor.attack import AttackScoringConfig
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    AdaptiveDispatchAttack,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selectors import (
    EpsilonGreedyTechniqueSelector,
    ContextExtractor,
    TechniqueSelector,
    global_context,
)

if TYPE_CHECKING:
    from pyrit.models import SeedAttackGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


class AdaptiveScenario(Scenario):
    """
    Abstract base for adaptive (epsilon-greedy) scenarios.

    Subclasses must implement the standard ``Scenario`` class-method overrides
    and declare ``VERSION`` and ``_atomic_attack_prefix``. Selector wiring,
    dispatcher construction, per-objective atomic-attack emission, and resume
    rehydration are handled here.
    """

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Enabled

    #: Subclasses must declare a scenario version for memory bookkeeping.
    VERSION: ClassVar[int]

    #: Prefix for per-objective atomic-attack names (e.g. ``"adaptive_text"``).
    _atomic_attack_prefix: ClassVar[str] = "adaptive"

    def __init__(
        self,
        *,
        objective_scorer: TrueFalseScorer | None = None,
        context_extractor: ContextExtractor = global_context,
        selector: TechniqueSelector | None = None,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Args:
            objective_scorer (TrueFalseScorer | None): Scorer used to judge each
                response. Defaults to the composite scorer from the base class.
            context_extractor (ContextExtractor): Maps a ``SeedAttackGroup`` to a
                context key. Defaults to ``global_context``.
            selector (TechniqueSelector | None): Pre-built selector. When ``None``
                (default) an :class:`EpsilonGreedyTechniqueSelector` is created
                with default settings.
            scenario_result_id (str | None): ID of an existing ``ScenarioResult`` to resume.
        """
        if not objective_scorer:
            objective_scorer = self._get_default_objective_scorer()
        self._objective_scorer: TrueFalseScorer = objective_scorer

        self._context_extractor = context_extractor
        self._custom_selector = selector

        super().__init__(
            version=self.VERSION,
            strategy_class=self.get_strategy_class(),
            objective_scorer=objective_scorer,
            scenario_result_id=scenario_result_id,
        )

    async def _get_atomic_attacks_async(self) -> list[AtomicAttack]:
        """
        Build one ``AtomicAttack`` per dataset, each carrying every objective
        in that dataset as a separate ``SeedAttackGroup``.

        A single ``AdaptiveDispatchAttack`` is constructed per dataset and
        shared across its seed groups; per-call seed-group routing and
        per-call ``seed_technique`` compatibility filtering happen inside the
        dispatcher (driven by ``AdaptiveDispatchParams.seed_group``). All
        dispatchers across all datasets share one ``TechniqueSelector``
        instance so learning accumulates globally.

        Seed groups whose objective is incompatible with every technique are
        dropped up-front with a warning so the dispatcher never sees an empty
        compatible pool at run time.

        Returns:
            list[AtomicAttack]: One ``AtomicAttack`` per dataset that has at
                least one compatible seed group. Empty if every seed group is
                incompatible with every selected technique.

        Raises:
            ValueError: If ``self._objective_target`` is not set, or if
                ``_build_techniques_dict`` finds no usable techniques.
        """
        if self._objective_target is None:
            raise ValueError("objective_target must be set before creating attacks")

        techniques = self._build_techniques_dict(objective_target=self._objective_target)

        selector: TechniqueSelector
        if self._custom_selector is not None:
            selector = self._custom_selector
        else:
            selector = EpsilonGreedyTechniqueSelector()
        # On resume, replay prior attempt outcomes from persisted metadata.
        self._rehydrate_selector_from_memory(selector=selector, known_techniques=set(techniques))

        seed_groups_by_dataset = self._dataset_config.get_seed_attack_groups()
        atomic_attacks: list[AtomicAttack] = []
        for dataset_name, seed_groups in seed_groups_by_dataset.items():
            atomic = self._build_atomic_for_dataset(
                dataset_name=dataset_name,
                seed_groups=seed_groups,
                techniques=techniques,
                selector=selector,
            )
            if atomic is not None:
                atomic_attacks.append(atomic)

        return atomic_attacks

    def _build_techniques_dict(
        self,
        *,
        objective_target: PromptTarget,
    ) -> dict[str, TechniqueBundle]:
        """
        Resolve selected strategies into a ``{name: TechniqueBundle}`` map.

        Each bundle carries the inner attack strategy along with the factory's
        ``seed_technique`` and ``adversarial_chat`` so the dispatcher can
        reproduce the static ``AtomicAttack`` execution path per attempt.

        Returns:
            dict[str, TechniqueBundle]: Mapping from technique name to its
                bundle, in the order selected strategies were resolved.

        Raises:
            ValueError: If no techniques remain after filtering. Includes the
                requested techniques and skip reasons.
        """
        selected_techniques = sorted({s.value for s in self._scenario_strategies})
        factories = self._get_attack_technique_factories()
        scoring_config = AttackScoringConfig(objective_scorer=self._objective_scorer)

        techniques: dict[str, TechniqueBundle] = {}
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
            techniques[technique_name] = TechniqueBundle(
                attack=technique.attack,
                seed_technique=technique.seed_technique,
                adversarial_chat=factory.adversarial_chat,
            )

        if not techniques:
            suffix = f" (skipped, no factory registered: {sorted(skipped_no_factory)})" if skipped_no_factory else ""
            raise ValueError(
                f"{type(self).__name__}: no usable techniques after resolving strategies. "
                f"Check the --strategies selection.{suffix}"
            )

        return techniques

    def _build_atomic_for_dataset(
        self,
        *,
        dataset_name: str,
        seed_groups: list[SeedAttackGroup],
        techniques: dict[str, TechniqueBundle],
        selector: TechniqueSelector,
    ) -> AtomicAttack | None:
        """
        Build a single ``AtomicAttack`` for one dataset with all compatible
        seed groups attached.

        Seed groups for which no technique in the pool is compatible are
        dropped here with a warning so the dispatcher's per-call compatible
        pool is guaranteed non-empty.

        Returns:
            AtomicAttack | None: The constructed atomic attack, or ``None`` when
                every seed group is incompatible with every technique.

        Raises:
            ValueError: If ``self._objective_target`` is not set (defensive
                guard; ``_get_atomic_attacks_async`` enforces this earlier).
        """
        if self._objective_target is None:  # pragma: no cover - defensive
            raise ValueError("objective_target must be set before creating attacks")

        compatible_seed_groups: list[SeedAttackGroup] = []
        for seed_group in seed_groups:
            has_compatible = any(
                bundle.seed_technique is None
                or seed_group.is_compatible_with_technique(technique=bundle.seed_technique)
                for bundle in techniques.values()
            )
            if has_compatible:
                compatible_seed_groups.append(seed_group)
            else:
                logger.warning(
                    "AdaptiveScenario: no compatible techniques for seed group in dataset '%s' "
                    "(objective=%r); skipping.",
                    dataset_name,
                    seed_group.objective.value,
                )

        if not compatible_seed_groups:
            return None

        dispatcher = AdaptiveDispatchAttack(
            objective_target=self._objective_target,
            techniques=techniques,
            selector=selector,
            context_extractor=self._context_extractor,
            objective_scorer=self._objective_scorer,
            max_attempts_per_objective=self.params["max_attempts_per_objective"],
        )

        return AtomicAttack(
            atomic_attack_name=f"{self._atomic_attack_prefix}_{dataset_name}",
            attack_technique=AttackTechnique(attack=dispatcher),
            seed_groups=compatible_seed_groups,
            objective_scorer=self._objective_scorer,
            memory_labels=dict(self._memory_labels),
            display_group=dataset_name,
        )

    def _rehydrate_selector_from_memory(
        self,
        *,
        selector: TechniqueSelector,
        known_techniques: set[str],
    ) -> None:
        """
        Replay persisted dispatch trails into ``selector`` so resume
        preserves learned state.

        Queries ``AttackResultEntry`` rows directly by ``scenario_result_id``
        (which selects on ``attribution_parent_id`` stamped at write time by
        ``AtomicAttack``'s attribution path) and filters to rows belonging to
        this scenario's adaptive atomic attacks via
        ``attribution_data["parent_collection"]``.

        Args:
            selector (TechniqueSelector): A freshly built selector to populate.
            known_techniques (set[str]): Techniques available in the current run.
                Trails referencing unknown techniques (e.g. after a strategies
                change) are skipped so replay can't poison the table.
        """
        if not self._scenario_result_id:
            return

        # Narrow to errors a memory backend would plausibly raise (DB/IO
        # failures, integrity issues). Programmer-level errors propagate.
        try:
            rows = self._memory.get_attack_results(scenario_result_id=self._scenario_result_id)
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(f"AdaptiveScenario: failed to load prior attack results for rehydration: {exc}")
            return

        adaptive_prefix = f"{self._atomic_attack_prefix}_"
        replayed = 0
        for result in rows:
            if result.attribution_data is None:
                continue
            collection = result.attribution_data.get("parent_collection")
            if not collection or not collection.startswith(adaptive_prefix):
                continue
            metadata = result.metadata or {}
            trail = metadata.get("adaptive_attempts")
            context = metadata.get("adaptive_context")
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
