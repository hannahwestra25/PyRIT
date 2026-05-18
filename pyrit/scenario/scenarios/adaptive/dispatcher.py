# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``AdaptiveDispatchAttack`` — an ``AttackStrategy`` that picks which inner
technique to run for each objective using an ``AdaptiveTechniqueSelector``.

This is the execution-side counterpart to the selector. The selector decides
*which arm to pull*; the dispatcher *runs the arm*, records the outcome, and
loops up to ``max_attempts_per_objective`` times.

The dispatcher reads a bandit-context key from
``context.memory_labels[BANDIT_CONTEXT_LABEL]``. The scenario is expected to
stamp that label per-objective (computed once at atomic-attack construction
time via a ``ContextExtractor``). When the label is missing, the global
context is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.core.attack_strategy import AttackContext, AttackStrategy
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


BANDIT_CONTEXT_LABEL: str = "_adaptive_context"
"""Memory-label key whose value is the bandit context string for an objective."""

ADAPTIVE_ARM_LABEL: str = "_adaptive_arm"
ADAPTIVE_ATTEMPT_LABEL: str = "_adaptive_attempt"


@dataclass
class AdaptiveDispatchContext(AttackContext[AttackParameters]):
    """
    Execution context for ``AdaptiveDispatchAttack``.

    No extra state is needed beyond what ``AttackContext`` provides; the
    dispatcher reads the objective and memory labels from the base class.
    """


class AdaptiveDispatchAttack(AttackStrategy[AdaptiveDispatchContext, AttackResult]):
    """
    Attack that delegates each attempt to one of several inner ``AttackStrategy``
    instances ("arms"), choosing per attempt via an ``AdaptiveTechniqueSelector``.

    For each objective the dispatcher loops up to ``max_attempts_per_objective``
    times. On each iteration it asks the selector which arm to try, executes
    the inner attack with the objective, records the outcome on the selector,
    and stops early on success.

    The selector instance is **shared by reference** with the scenario, so
    learning accumulates across all objectives in a run.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        arms: dict[str, AttackStrategy[Any, AttackResult]],
        selector: AdaptiveTechniqueSelector,
        max_attempts_per_objective: int = 3,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target the inner attacks run against.
                Stored for identifier/logging parity; the dispatcher does not call
                the target directly.
            arms (dict[str, AttackStrategy[Any, AttackResult]]): Mapping from
                technique name to a pre-built inner attack. Must be non-empty.
            selector (AdaptiveTechniqueSelector): Shared bandit state.
            max_attempts_per_objective (int): Maximum number of arm attempts
                per objective. Must be >= 1. Defaults to 3.

        Raises:
            ValueError: If ``arms`` is empty or ``max_attempts_per_objective`` < 1.
        """
        if not arms:
            raise ValueError("arms must contain at least one technique")
        if max_attempts_per_objective < 1:
            raise ValueError(f"max_attempts_per_objective must be >= 1, got {max_attempts_per_objective}")

        super().__init__(
            objective_target=objective_target,
            context_type=AdaptiveDispatchContext,
            params_type=AttackParameters,
            logger=logger,
        )
        self._arms = arms
        self._selector = selector
        self._max_attempts = max_attempts_per_objective

    def _validate_context(self, *, context: AdaptiveDispatchContext) -> None:
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty")

    async def _setup_async(self, *, context: AdaptiveDispatchContext) -> None:
        pass

    async def _teardown_async(self, *, context: AdaptiveDispatchContext) -> None:
        pass

    async def _perform_async(self, *, context: AdaptiveDispatchContext) -> AttackResult:
        bandit_context = context.memory_labels.get(BANDIT_CONTEXT_LABEL, GLOBAL_CONTEXT)
        arm_names = list(self._arms.keys())

        last_result: AttackResult | None = None
        trail: list[dict[str, str]] = []

        for attempt_idx in range(self._max_attempts):
            chosen = self._selector.select(context=bandit_context, arms=arm_names)
            inner = self._arms[chosen]
            attempt_labels = {
                **context.memory_labels,
                ADAPTIVE_ARM_LABEL: chosen,
                ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
            }

            logger.debug(
                "AdaptiveDispatchAttack: attempt %d/%d context=%r arm=%r",
                attempt_idx + 1,
                self._max_attempts,
                bandit_context,
                chosen,
            )

            result = await inner.execute_async(
                objective=context.objective,
                memory_labels=attempt_labels,
            )
            success = result.outcome == AttackOutcome.SUCCESS
            self._selector.update(context=bandit_context, technique=chosen, success=success)

            trail.append({"technique": chosen, "outcome": result.outcome.value})
            last_result = result

            if success:
                break

        # ``max_attempts`` is validated >= 1 above, so the loop always runs at least once.
        assert last_result is not None
        last_result.metadata = {
            **last_result.metadata,
            "adaptive_attempts": trail,
            "adaptive_context": bandit_context,
        }
        return last_result
