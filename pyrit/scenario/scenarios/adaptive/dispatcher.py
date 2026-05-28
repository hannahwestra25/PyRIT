# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveDispatchAttack`` — picks inner techniques per objective via a
``TechniqueSelector``, then runs them in priority order via a
``SequentialAttack`` (stop on first success).

The selector is stateless and async: it queries memory for historical
success rates. The dispatcher pre-selects up to ``max_attempts_per_objective``
techniques at the start of each objective, builds a per-call
``SequentialAttack`` whose child attacks are the chosen techniques, and
delegates iteration + stop-on-success + envelope construction to that
compound attack.

Returned envelope is a ``SequentialAttackResult`` (an ``AttackResult``
subclass) stamped with the adaptive trail under
``metadata["adaptive_attempts"]``. Inner per-attempt results live on the
envelope's ``child_attack_results`` and persist as their own DB rows; the
envelope itself owns no conversation (``conversation_id == ""``).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from pyrit.executor.attack.compound.sequential_attack import (
    SequenceCompletionPolicy,
    SequentialAttack,
    SequentialAttackResult,
    SequentialChildAttack,
)
from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.core.attack_strategy import AttackContext, AttackStrategy
from pyrit.scenario.scenarios.adaptive.selectors.technique_selector import ADAPTIVE_TECHNIQUE_LABEL

if TYPE_CHECKING:
    from pyrit.models import AttackResult, SeedAttackGroup, SeedAttackTechniqueGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.scenarios.adaptive.selectors import TechniqueSelector
    from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


# Memory-label keys stamped onto persisted prompt rows so adaptive attempts
# can be filtered/grouped after a run.
ADAPTIVE_ATTEMPT_LABEL: str = "_adaptive_attempt"
"""1-based attempt index within the per-objective loop."""


@dataclass(frozen=True)
class TechniqueBundle:
    """
    Per-technique bundle consumed by the dispatcher.

    Carries the inner attack strategy alongside the factory-supplied
    ``seed_technique`` (if any) and ``adversarial_chat`` (required when the
    seed_technique contains a simulated-conversation config).
    """

    attack: AttackStrategy[Any, AttackResult]
    name: str = ""
    seed_technique: SeedAttackTechniqueGroup | None = None
    adversarial_chat: PromptTarget | None = None


@dataclass(frozen=True)
class AdaptiveDispatchParams(AttackParameters):
    """Attack parameters for adaptive dispatch, carrying the original seed group."""

    # The original SeedAttackGroup is preserved on the params so the
    # dispatcher can apply per-attempt seed_technique merging and derive
    # the per-call adaptive context. Captured by ``from_seed_group_async``;
    # not user-supplied via overrides.
    seed_group: Optional[SeedAttackGroup] = field(default=None, repr=False, compare=False)

    @classmethod
    async def from_seed_group_async(
        cls,
        *,
        seed_group: SeedAttackGroup,
        adversarial_chat: Optional[PromptTarget] = None,  # noqa: ARG003 — required by base class signature
        objective_scorer: Optional[TrueFalseScorer] = None,  # noqa: ARG003 — required by base class signature
        **overrides: Any,
    ) -> AdaptiveDispatchParams:
        """
        Build params for a single dispatch and capture the original seed_group.

        The dispatcher applies seed_technique merging itself per-attempt
        (when constructing the per-call ``SequentialChildAttack``s), so we
        deliberately bypass the base class's simulated-conversation
        expansion / next_message extraction: each inner technique runs
        through its own ``AttackExecutor`` call inside ``SequentialAttack``
        which performs that work using the technique-merged seed_group.

        Returns:
            AdaptiveDispatchParams: The constructed parameters with the seed group attached.

        Raises:
            ValueError: If the seed_group's objective is not initialized or invalid overrides are passed.
        """
        if seed_group.objective is None:
            raise ValueError("seed_group.objective is not initialized")
        seed_group.validate()

        valid_fields = {f.name for f in dataclasses.fields(cls)} - {"seed_group"}
        invalid = set(overrides.keys()) - valid_fields
        if invalid:
            raise ValueError(f"{cls.__name__} does not accept parameters: {invalid}. Accepted: {valid_fields}")

        return cls(
            objective=seed_group.objective.value,
            memory_labels=overrides.get("memory_labels") or {},
            seed_group=seed_group,
        )


@dataclass
class AdaptiveDispatchContext(AttackContext[AdaptiveDispatchParams]):
    """Execution context for ``AdaptiveDispatchAttack`` (no extra state)."""


class AdaptiveDispatchAttack(AttackStrategy[AdaptiveDispatchContext, SequentialAttackResult]):
    """
    Attack that delegates each attempt to one of several inner techniques,
    choosing per attempt via a ``TechniqueSelector``.

    For each objective: query the selector for the top
    ``max_attempts_per_objective`` techniques compatible with the seed
    group, then hand the chosen techniques off to a per-call
    ``SequentialAttack`` (with ``SequenceCompletionPolicy.FIRST_SUCCESS``)
    which iterates through them, stops on the first success, and returns
    one ``SequentialAttackResult`` envelope. The selector is shared by
    reference across all dispatch calls in a scenario so learning
    accumulates across objectives.

    The seed group for a given dispatch is read from
    ``context.params.seed_group`` (captured by
    ``AdaptiveDispatchParams.from_seed_group_async``). When a chosen
    technique declares a ``seed_technique``, that group is merged into the
    seed group when building the per-attempt ``SequentialChildAttack``
    (mirroring the static ``AtomicAttack`` path). Techniques whose
    ``seed_technique`` is incompatible with the current seed group are
    filtered out of the candidate pool for that call; if the pool is empty
    the dispatcher raises so the per-call seed group is dropped by the
    executor's partial-failure path rather than silently no-op'ing.

    The returned envelope owns no conversation of its own
    (``conversation_id == ""``): the inner per-attempt ``AttackResult``s
    each persist their own row with the raw conversation, and are
    reachable via ``result.child_attack_results`` (in-memory) or via
    ``result.child_attack_result_ids`` (after a DB round-trip). The
    envelope itself is persisted with the dispatch trail stamped onto
    ``metadata["adaptive_attempts"]``.
    """

    ADAPTIVE_ATTEMPTS_KEY: str = "adaptive_attempts"
    """Metadata key under which the per-attempt dispatch trail is stamped on the envelope."""

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        techniques: dict[str, TechniqueBundle],
        selector: TechniqueSelector,
        objective_scorer: TrueFalseScorer | None = None,
        max_attempts_per_objective: int = 3,
        scenario_result_id: str | None = None,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target inner attacks run against.
            techniques (dict[str, TechniqueBundle]): Mapping from technique eval hash to
                its bundle (attack, name, seed_technique, adversarial_chat). Must be non-empty.
            selector (TechniqueSelector): Stateless technique selector.
            objective_scorer (TrueFalseScorer | None): Scorer passed through to
                techniques that generate simulated conversations.
            max_attempts_per_objective (int): Max attempts per objective; >= 1.
                Defaults to 3.
            scenario_result_id (str | None): If provided, passed to the selector
                to scope memory queries to this scenario run. Defaults to ``None``.

        Raises:
            ValueError: If ``techniques`` is empty or ``max_attempts_per_objective`` < 1.
        """
        if not techniques:
            raise ValueError("techniques must contain at least one attack technique")
        if max_attempts_per_objective < 1:
            raise ValueError(f"max_attempts_per_objective must be >= 1, got {max_attempts_per_objective}")

        super().__init__(
            objective_target=objective_target,
            context_type=AdaptiveDispatchContext,
            params_type=AdaptiveDispatchParams,
            logger=logger,
        )
        self._techniques = techniques
        self._selector = selector
        self._objective_scorer = objective_scorer
        self._max_attempts = max_attempts_per_objective
        self._scenario_result_id = scenario_result_id

    def _validate_context(self, *, context: AdaptiveDispatchContext) -> None:
        """
        Ensure the context carries a non-empty objective string.

        Raises:
            ValueError: If ``context.objective`` is empty or whitespace-only.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty")

    async def _setup_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt setup is owned by ``SequentialAttack`` / its executor."""

    async def _teardown_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt teardown is owned by ``SequentialAttack`` / its executor."""

    def _build_child_attacks(
        self,
        *,
        seed_group: SeedAttackGroup,
        chosen_techniques: list[str],
    ) -> list[SequentialChildAttack]:
        """
        Build the ``SequentialChildAttack`` list handed to ``SequentialAttack``.

        Per chosen technique: merge ``bundle.seed_technique`` into
        ``seed_group`` (if any), and stamp the per-attempt
        ``ADAPTIVE_TECHNIQUE_LABEL`` and ``ADAPTIVE_ATTEMPT_LABEL`` memory
        labels. ``SequentialAttack`` further merges these with the
        compound's own ``context.memory_labels`` at dispatch time, so the
        outer caller's labels still propagate to every attempt.

        Args:
            seed_group (SeedAttackGroup): The seed group for this dispatch call.
            chosen_techniques (list[str]): Technique eval hashes returned by
                the selector, in priority order.

        Returns:
            list[SequentialChildAttack]: One child attack per chosen
                technique, in dispatch order.
        """
        child_attacks: list[SequentialChildAttack] = []
        for attempt_idx, chosen in enumerate(chosen_techniques):
            bundle = self._techniques[chosen]
            execution_group = (
                seed_group.with_technique(technique=bundle.seed_technique)
                if bundle.seed_technique is not None
                else seed_group
            )
            child_attacks.append(
                SequentialChildAttack(
                    strategy=bundle.attack,
                    seed_group=execution_group,
                    adversarial_chat=bundle.adversarial_chat,
                    objective_scorer=self._objective_scorer,
                    memory_labels={
                        ADAPTIVE_TECHNIQUE_LABEL: chosen,
                        ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
                    },
                )
            )
        return child_attacks

    def _build_adaptive_trail(
        self,
        *,
        chosen_techniques: list[str],
        child_results: list[AttackResult],
    ) -> list[dict[str, str]]:
        """
        Build the per-attempt dispatch trail stamped onto the envelope.

        ``chosen_techniques`` is the full pre-selected list; ``child_results``
        contains only the attempts that actually ran (``FIRST_SUCCESS`` may
        halt early). Trail length matches ``len(child_results)``.

        Returns:
            list[dict[str, str]]: One entry per executed attempt, in
                dispatch order, each carrying ``technique`` (display name),
                ``technique_hash`` (eval hash), and ``outcome``
                (``AttackOutcome`` value).
        """
        return [
            {
                "technique": self._techniques[h].name,
                "technique_hash": h,
                "outcome": r.outcome.value,
            }
            for h, r in zip(chosen_techniques, child_results, strict=False)
        ]

    async def _perform_async(self, *, context: AdaptiveDispatchContext) -> SequentialAttackResult:
        """
        Run the per-objective adaptive loop via ``SequentialAttack``.

        Queries the stateless selector for the top
        ``max_attempts_per_objective`` techniques (filtered by per-call
        seed-group compatibility), wraps them in a ``SequentialAttack``
        with ``SequenceCompletionPolicy.FIRST_SUCCESS``, and delegates
        iteration + stop-on-success + envelope construction. Stamps the
        ``adaptive_attempts`` trail on the envelope before returning.

        Args:
            context (AdaptiveDispatchContext): Execution context whose
                ``params.seed_group`` carries the seed group for this call.

        Returns:
            SequentialAttackResult: The envelope produced by the inner
                ``SequentialAttack`` with the adaptive trail stamped onto
                ``metadata["adaptive_attempts"]``.

        Raises:
            ValueError: If ``context.params.seed_group`` is missing, or if no
                techniques in the pool are compatible with the seed group.
        """
        seed_group = context.params.seed_group
        if seed_group is None:
            raise ValueError(
                "AdaptiveDispatchAttack requires AdaptiveDispatchParams.seed_group; "
                "build params via AdaptiveDispatchParams.from_seed_group_async."
            )

        compatible_names = [
            name
            for name, bundle in self._techniques.items()
            if bundle.seed_technique is None or seed_group.is_compatible_with_technique(technique=bundle.seed_technique)
        ]
        if not compatible_names:
            raise ValueError(
                f"AdaptiveDispatchAttack: no compatible techniques for seed group "
                f"(objective={seed_group.objective.value!r})."
            )

        chosen_techniques = await self._selector.select_async(
            technique_identifiers=compatible_names,
            objective=context.objective,
            num_top_techniques=self._max_attempts,
            scenario_result_id=self._scenario_result_id,
        )

        child_attacks = self._build_child_attacks(
            seed_group=seed_group,
            chosen_techniques=chosen_techniques,
        )

        sequential = SequentialAttack(
            objective_target=self._objective_target,
            child_attacks=child_attacks,
            completion_policy=SequenceCompletionPolicy.FIRST_SUCCESS,
        )

        result: SequentialAttackResult = await sequential.execute_async(
            objective=context.objective,
            memory_labels=dict(context.memory_labels),
        )

        result.metadata[self.ADAPTIVE_ATTEMPTS_KEY] = self._build_adaptive_trail(
            chosen_techniques=chosen_techniques,
            child_results=result.child_attack_results,
        )
        return result
