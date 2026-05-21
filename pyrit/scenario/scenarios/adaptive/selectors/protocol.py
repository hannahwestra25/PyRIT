# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Selector protocol and context extractors for adaptive scenarios."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup

ContextExtractor = Callable[["SeedAttackGroup"], str]
"""Maps a ``SeedAttackGroup`` to an adaptive context key."""

GLOBAL_CONTEXT: str = "_global"
"""Default context: all objectives share one selection table."""

UNCATEGORIZED_CONTEXT: str = "_uncategorized"
"""Fallback context for seed groups with no harm category metadata."""


def global_context(_seed_attack_group: SeedAttackGroup) -> str:
    """
    Return a single shared context for all objectives.

    Returns:
        str: Always :data:`GLOBAL_CONTEXT`.
    """
    return GLOBAL_CONTEXT


def harm_category_context(seed_attack_group: SeedAttackGroup) -> str:
    """
    Return a context keyed by the sorted, ``|``-joined harm categories.

    Multi-category seeds form their own bucket; sorting makes the key deterministic.

    Returns:
        str: The ``|``-joined sorted harm categories, or :data:`UNCATEGORIZED_CONTEXT`
            when the seed group has no categories.
    """
    categories = seed_attack_group.harm_categories
    if not categories:
        return UNCATEGORIZED_CONTEXT
    return "|".join(sorted(categories))


@runtime_checkable
class TechniqueSelector(Protocol):
    """
    Protocol for adaptive technique selectors.

    Any object implementing ``select`` and ``record_outcome`` can serve as
    the selector for an ``AdaptiveScenario``. The epsilon-greedy
    implementation (:class:`EpsilonGreedyTechniqueSelector`) is the default.
    """

    def select(self, *, context: str, techniques: Sequence[str], decision_key: str = "") -> str:
        """Pick the next technique to try for ``context``."""
        ...  # pragma: no cover

    def record_outcome(self, *, context: str, technique: str, success: bool) -> None:
        """Record the outcome of an attempt."""
        ...  # pragma: no cover
