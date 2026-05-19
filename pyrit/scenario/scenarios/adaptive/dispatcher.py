# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""``AdaptiveDispatchAttack`` — picks an inner technique per attempt via an
``AdaptiveTechniqueSelector``, runs it, records the outcome, and loops up to
``max_attempts_per_objective`` times. Reads the per-objective context key from
``context.memory_labels[ADAPTIVE_CONTEXT_LABEL]`` (falls back to the global context).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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


# Memory-label keys stamped onto persisted prompt rows so adaptive attempts
# can be filtered/grouped after a run. The scenario stamps the context once
# per objective; the dispatcher stamps technique + attempt index on each try.
ADAPTIVE_CONTEXT_LABEL: str = "_adaptive_context"
"""Per-objective context key (e.g. ``"_global"`` or a harm category)."""
ADAPTIVE_TECHNIQUE_LABEL: str = "_adaptive_technique"
"""Technique chosen by the dispatcher for a given attempt."""
ADAPTIVE_ATTEMPT_LABEL: str = "_adaptive_attempt"
"""1-based attempt index within the per-objective loop."""


@dataclass
class AdaptiveDispatchContext(AttackContext[AttackParameters]):
    """Execution context for ``AdaptiveDispatchAttack`` (no extra state)."""


class AdaptiveDispatchAttack(AttackStrategy[AdaptiveDispatchContext, AttackResult]):
    """Attack that delegates each attempt to one of several inner techniques,
    choosing per attempt via an ``AdaptiveTechniqueSelector``.

    For each objective, loops up to ``max_attempts_per_objective`` times:
    ask the selector, execute the chosen technique, record the outcome, and
    stop early on success. The selector is shared by reference with the
    scenario so learning accumulates across objectives.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        techniques: dict[str, AttackStrategy[Any, AttackResult]],
        selector: AdaptiveTechniqueSelector,
        max_attempts_per_objective: int = 3,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target inner attacks run against.
                Stored for identifier/logging parity; not called directly.
            techniques (dict[str, AttackStrategy[Any, AttackResult]]): Mapping from
                technique name to a pre-built inner attack. Must be non-empty.
            selector (AdaptiveTechniqueSelector): Shared selector state.
            max_attempts_per_objective (int): Max attempts per objective; >= 1.
                Defaults to 3.

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
            params_type=AttackParameters,
            logger=logger,
        )
        self._techniques = techniques
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
        adaptive_context = context.memory_labels.get(ADAPTIVE_CONTEXT_LABEL, GLOBAL_CONTEXT)
        technique_names = list(self._techniques.keys())

        last_result: AttackResult | None = None
        trail: list[dict[str, str]] = []

        for attempt_idx in range(self._max_attempts):
            chosen = self._selector.select(context=adaptive_context, techniques=technique_names)
            inner = self._techniques[chosen]
            attempt_labels = {
                **context.memory_labels,
                ADAPTIVE_TECHNIQUE_LABEL: chosen,
                ADAPTIVE_ATTEMPT_LABEL: str(attempt_idx + 1),
            }

            logger.debug(
                "AdaptiveDispatchAttack: attempt %d/%d context=%r technique=%r",
                attempt_idx + 1,
                self._max_attempts,
                adaptive_context,
                chosen,
            )

            result = await inner.execute_async(
                objective=context.objective,
                memory_labels=attempt_labels,
            )
            success = result.outcome == AttackOutcome.SUCCESS
            self._selector.record_outcome(context=adaptive_context, technique=chosen, success=success)

            trail.append({"technique": chosen, "outcome": result.outcome.value})
            last_result = result

            if success:
                break

        # ``max_attempts`` is validated >= 1, so the loop always runs at least
        # once. Guard explicitly rather than with ``assert`` (stripped under -O).
        if last_result is None:  # pragma: no cover - defensive
            raise RuntimeError("AdaptiveDispatchAttack ran zero attempts; this should be unreachable.")
        # Return a fresh dispatcher-owned ``AttackResult``: the inner attack
        # already persisted ``last_result`` via its own post-execute hook, so
        # returning it directly would cause a PK conflict on the outer hook.
        # ``dataclasses.replace`` copies every field; we override identity
        # fields and stamp the trail onto metadata.
        return replace(
            last_result,
            attack_result_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={
                **last_result.metadata,
                "adaptive_attempts": trail,
                "adaptive_context": adaptive_context,
            },
        )
