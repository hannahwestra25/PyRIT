# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Technique selector protocol for adaptive scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable


# TODO: probably want to expand this to allow for more filtering options
# (e.g. filter by scenario parameters, attack labels, etc.)
class SelectorScope(str, Enum):
    """Controls which historical data a selector queries."""

    ALL_RUNS = "all_runs"
    """Use technique success rates from all historical scenario runs."""

    CURRENT_RUN = "current_run"
    """Use technique success rates only from the current scenario run."""


ADAPTIVE_TECHNIQUE_LABEL: str = "_adaptive_technique"
"""Memory-label key the dispatcher stamps on each attack result to record
which technique was used."""


@runtime_checkable
class TechniqueSelector(Protocol):
    """
    Protocol for adaptive technique selectors.

    Selectors are **stateless** — they query memory for historical success
    rates rather than maintaining internal counts. Calling ``select_async``
    with the same arguments twice should yield the same answer
    (deterministic given memory contents).
    """

    async def select_async(
        self,
        *,
        technique_identifiers: Sequence[str],
        objective: str,
        num_top_techniques: int = 1,
        scenario_result_id: str | None = None,
    ) -> Sequence[str]:
        """
        Return techniques in priority order (try first, try second, …).

        Args:
            technique_identifiers (Sequence[str]): Available technique names.
            objective (str): The objective text for this selection.
            num_top_techniques (int): Max techniques to return. Defaults to 1.
            scenario_result_id (str | None): The current scenario run ID,
                provided by the dispatcher. Selectors use this when their
                scope is ``SelectorScope.CURRENT_RUN``.

        Returns:
            Sequence[str]: Up to ``num_top_techniques`` technique names in
                priority order. Fewer if not enough techniques are available.
        """
        ...  # pragma: no cover
