# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveDispatchAttack`` — picks inner techniques per objective via a
``TechniqueSelector``, runs them in priority order, and stops on success.

The selector is stateless and async: it queries memory for historical
success rates. The dispatcher pre-selects up to ``max_attempts_per_objective``
techniques at the start of each objective, then iterates through them.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from pyrit.executor.attack.core.attack_executor import AttackExecutor
from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.core.attack_strategy import AttackContext, AttackStrategy
from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup
from pyrit.scenario.scenarios.adaptive.selectors.technique_selector import ADAPTIVE_TECHNIQUE_LABEL

if TYPE_CHECKING:
    from pyrit.models import SeedAttackTechniqueGroup
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

        The dispatcher applies seed_technique merging itself per-attempt, so
        we deliberately bypass the base class's simulated-conversation
        expansion / next_message extraction: the inner technique runs through
        its own ``execute_attack_from_seed_groups_async`` call which performs
        that work using the technique-merged seed_group.

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


class AdaptiveDispatchAttack(AttackStrategy[AdaptiveDispatchContext, AttackResult]):
    """
    Attack that delegates each attempt to one of several inner techniques,
    choosing per attempt via a ``TechniqueSelector``.

    For each objective, loops up to ``max_attempts_per_objective`` times:
    ask the selector, execute the chosen technique against the current seed
    group, record the outcome, and stop early on success. The selector is
    shared by reference across all dispatch calls in a scenario so learning
    accumulates across objectives.

    The seed group for a given dispatch is read from
    ``context.params.seed_group`` (captured by
    ``AdaptiveDispatchParams.from_seed_group_async``). When a chosen
    technique declares a ``seed_technique``, that group is merged into the
    seed group before execution (mirroring the static ``AtomicAttack`` path).
    Techniques whose ``seed_technique`` is incompatible with the current
    seed group are filtered out of the candidate pool for that call; if the
    pool is empty the dispatcher raises so the per-call seed group is dropped
    by the executor's partial-failure path rather than silently no-op'ing.

    On success, the dispatcher returns a fresh ``AttackResult`` copy of the
    winning inner result (new ``attack_result_id`` and ``timestamp``) with
    the dispatch trail stamped onto ``metadata``. The inner result has
    already been persisted by its own post-execute hook, so two rows are
    written per successful objective sharing the same ``conversation_id``:
    the inner row carries the raw outcome, the outer row carries the
    adaptive trail.
    """

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
        self._executor = AttackExecutor(max_concurrency=1)

    def _validate_context(self, *, context: AdaptiveDispatchContext) -> None:
        """
        Ensure the context carries a non-empty objective string.

        Raises:
            ValueError: If ``context.objective`` is empty or whitespace-only.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty")

    async def _setup_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt setup is owned by the inner technique's executor."""

    async def _teardown_async(self, *, context: AdaptiveDispatchContext) -> None:
        """No-op: per-attempt teardown is owned by the inner technique's executor."""

    async def _run_inner_attack_async(
        self,
        *,
        bundle: TechniqueBundle,
        seed_group: SeedAttackGroup,
        attempt_labels: dict[str, str],
    ) -> AttackResult:
        """
        Execute the chosen technique against the per-call seed group.

        Merges ``bundle.seed_technique`` into ``seed_group`` (when present)
        and delegates execution to ``AttackExecutor``. Isolated as a method
        so tests can patch the inner-attack call surface.

        Args:
            bundle (TechniqueBundle): The chosen technique's attack + seeds + chat.
            seed_group (SeedAttackGroup): The seed group for this dispatch call.
            attempt_labels (dict[str, str]): Memory labels stamped onto this attempt.

        Returns:
            AttackResult: The single result produced for this attempt.

        Raises:
            RuntimeError: If the executor returned no completed results and no
                propagated exception (should be unreachable).
        """
        if bundle.seed_technique is not None:
            execution_group = seed_group.with_technique(technique=bundle.seed_technique)
        else:
            execution_group = seed_group

        executor_result = await self._executor.execute_attack_from_seed_groups_async(
            attack=bundle.attack,
            seed_groups=[execution_group],
            adversarial_chat=bundle.adversarial_chat,
            objective_scorer=self._objective_scorer,
            memory_labels=attempt_labels,
        )

        if executor_result.completed_results:
            return executor_result.completed_results[0]
        if executor_result.incomplete_objectives:
            raise executor_result.incomplete_objectives[0][1]
        raise RuntimeError(  # pragma: no cover - defensive
            "AttackExecutor returned neither completed nor incomplete results."
        )

    async def _perform_async(self, *, context: AdaptiveDispatchContext) -> AttackResult:
        """
        Run the per-objective adaptive loop.

        Pre-selects up to ``max_attempts_per_objective`` techniques via the
        stateless selector, then iterates in priority order. Stops early on
        success.

        Args:
            context (AdaptiveDispatchContext): Execution context whose
                ``params.seed_group`` carries the seed group for this call.

        Returns:
            AttackResult: A fresh dispatcher-owned copy of the final inner
                result with the dispatch trail stamped onto ``metadata``.

        Raises:
            ValueError: If ``context.params.seed_group`` is missing, or if no
                techniques in the pool are compatible with the seed group.
            RuntimeError: If the loop somehow ran zero attempts (unreachable
                because ``max_attempts_per_objective`` is validated >= 1).
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

        last_result: AttackResult | None = None
        trail: list[dict[str, str]] = []

        for attempt_idx, chosen in enumerate(chosen_techniques):
            bundle = self._techniques[chosen]
            attempt_labels = {
                **context.memory_labels,
                ADAPTIVE_TECHNIQUE_LABEL: chosen,
                ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
            }

            logger.debug(
                "AdaptiveDispatchAttack: attempt %d/%d technique=%r (hash=%s)",
                attempt_idx + 1,
                len(chosen_techniques),
                bundle.name,
                chosen,
            )

            result = await self._run_inner_attack_async(
                bundle=bundle, seed_group=seed_group, attempt_labels=attempt_labels
            )

            trail.append({"technique": bundle.name, "technique_hash": chosen, "outcome": result.outcome.value})
            last_result = result

            if result.outcome == AttackOutcome.SUCCESS:
                break

        if last_result is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaptiveDispatchAttack ran zero attempts; this should be unreachable.")
        return replace(
            last_result,
            attack_result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={
                **last_result.metadata,
                "adaptive_attempts": trail,
            },
        )
