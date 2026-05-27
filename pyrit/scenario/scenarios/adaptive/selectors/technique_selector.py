# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Technique selector protocol for adaptive scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class SelectorScope:
    """
    Filter describing which historical ``AttackResult`` rows a selector
    queries when estimating technique success rates.

    All fields default to "no restriction"; combine fields to narrow the
    scope (e.g. current run only, same scenario class, same harm category).
    Filter values flow through :func:`compute_technique_success_rates` to
    :meth:`MemoryInterface.get_attack_results`.

    The scope is held by the selector at construction time. The per-call
    ``scenario_result_id`` is supplied by the dispatcher and is forwarded
    to memory only when ``current_run_only`` is set; otherwise the selector
    queries across all runs.
    """

    current_run_only: bool = False
    """Restrict to the dispatcher-supplied ``scenario_result_id`` for the
    in-flight run. When ``False`` (default), query across all runs."""

    attack_classes: Sequence[str] | None = None
    """Filter to results emitted by these attack / scenario class names
    (e.g. ``["TextAdaptive"]``). Useful to keep one modality's bandit from
    being influenced by another's history. ``None`` means no class filter."""

    targeted_harm_categories: Sequence[str] | None = None
    """Filter to results whose prompts targeted these harm categories.
    ``None`` means no harm-category filter."""

    extra_labels: Mapping[str, str | Sequence[str]] | None = None
    """Additional memory-label filters merged on top of the technique-hash
    label that the selector adds internally. Use this as an escape hatch
    for label-based filtering not covered by the named fields."""

    @classmethod
    def all_runs(cls) -> SelectorScope:
        """
        Build a scope that queries across all historical scenario runs (the default).

        Returns:
            SelectorScope: A scope with no restrictions.
        """
        return cls()

    @classmethod
    def current_run(cls) -> SelectorScope:
        """
        Build a scope restricted to the dispatcher-supplied scenario run.

        Returns:
            SelectorScope: A scope with ``current_run_only=True``.
        """
        return cls(current_run_only=True)


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
                provided by the dispatcher. Selectors forward this to
                memory only when their :class:`SelectorScope` has
                ``current_run_only=True``.

        Returns:
            Sequence[str]: Up to ``num_top_techniques`` technique names in
                priority order. Fewer if not enough techniques are available.
        """
        ...  # pragma: no cover
