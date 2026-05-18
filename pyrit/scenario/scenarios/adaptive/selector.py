# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Adaptive technique selection for the ``TextAdaptive`` scenario.

This module provides:
    - ``AdaptiveTechniqueSelector``: an epsilon-greedy selector keyed by
      ``(context, technique)`` that tracks successes/attempts per technique and
      picks the next technique to try.
    - ``ContextExtractor``: a callable alias for deriving a context string
      from a ``SeedAttackGroup``, plus two ready-made extractors:
      ``global_context`` (single bucket) and ``harm_category_context``
      (first harm category, falling back to ``"_uncategorized"``).

The selector is intentionally I/O-free and synchronous; it holds a small
mutable table that lives for the duration of a single scenario run.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup


ContextExtractor = Callable[["SeedAttackGroup"], str]
"""Maps a ``SeedAttackGroup`` to an adaptive context key (e.g. a harm category)."""


# Sentinel context keys used when no per-objective partitioning is desired
# or when a seed group lacks harm category metadata.
GLOBAL_CONTEXT: str = "_global"
"""Default context key: all objectives share one selection table."""
UNCATEGORIZED_CONTEXT: str = "_uncategorized"
"""Fallback context for seed groups with no harm category metadata."""


# Context extractors are module-level functions so they can be passed directly
# as the ``context_extractor`` argument to ``TextAdaptive``. They implement the
# ``ContextExtractor`` callable protocol.


def global_context(_seed_attack_group: SeedAttackGroup) -> str:
    """Return a constant context so all objectives share one selection table."""
    return GLOBAL_CONTEXT


def harm_category_context(seed_attack_group: SeedAttackGroup) -> str:
    """Return the first harm category on the seed group, or a fallback."""
    categories = seed_attack_group.harm_categories
    if not categories:
        return UNCATEGORIZED_CONTEXT
    return categories[0]


class AdaptiveTechniqueSelector:
    """
    Epsilon-greedy selector over attack techniques.

    The selector maintains a table of ``(context, technique) -> (successes, attempts)``
    counts. ``select`` returns the next technique to try for a given context,
    and ``record_outcome`` records the outcome of an attempt.

    Selection uses epsilon-greedy with optimistic initialization:
        - With probability ``epsilon``, pick uniformly at random from ``techniques``.
        - Otherwise, pick the technique with the highest estimated success rate.
          The estimate is ``(successes + 1) / (attempts + 1)`` (Laplace smoothing),
          so unseen techniques start at 100% and are explored first via tiebreak.

    When a ``(context, technique)`` cell has fewer than ``pool_threshold`` attempts,
    the estimate falls back to the pooled global rate for that technique across all
    contexts. This lets per-context selectors benefit from cross-context data
    until they have enough local samples. Set ``pool_threshold=1`` to disable
    pooling (use the local estimate as soon as any attempt is recorded).

    Note:
        This class is not thread/async safe. It assumes sequential calls,
        which matches the base ``Scenario._execute_scenario_async`` loop
        (same pattern as all other scenarios).
    """

    # Tolerance for floating-point comparison when tiebreaking in exploitation.
    # Current estimates are exact rationals, but this guards against future
    # estimator changes that may introduce floating-point drift.
    _TIE_TOL: float = 1e-12

    def __init__(
        self,
        *,
        epsilon: float = 0.2,
        pool_threshold: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        """
        Args:
            epsilon (float): Exploration probability in [0.0, 1.0]. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before
                the local estimate replaces the pooled-global estimate. Until this
                threshold is reached, the selector uses the technique's average
                across all contexts. Must be >= 1; set to 1 to disable pooling.
                Defaults to 3.
            rng (random.Random | None): A ``random.Random`` instance for
                reproducible selection decisions. Using a dedicated RNG (rather
                than a bare float) enables seeded determinism across the full
                sequence of select calls within a run. Defaults to a fresh
                unseeded ``random.Random()``.

        Raises:
            ValueError: If ``epsilon`` is outside [0.0, 1.0] or
                ``pool_threshold`` is < 1.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        if pool_threshold < 1:
            raise ValueError(f"pool_threshold must be >= 1, got {pool_threshold}")

        self._epsilon = epsilon
        self._pool_threshold = pool_threshold
        self._rng = rng if rng is not None else random.Random()
        self._counts: dict[tuple[str, str], tuple[int, int]] = {}
        # Per-arm pooled counts, kept in sync with ``_counts`` in ``update`` so
        # ``_estimate``'s pooled-backoff branch is O(1).
        self._global_counts: dict[str, tuple[int, int]] = {}

    def select(self, *, context: str, techniques: Sequence[str]) -> str:
        """
        Pick the next technique to try for ``context``.

        Args:
            context (str): The context key (e.g. ``"_global"`` or a harm category).
            techniques (Sequence[str]): The candidate technique names.

        Returns:
            str: The chosen technique name.

        Raises:
            ValueError: If ``techniques`` is empty.
        """
        technique_list = list(techniques)
        if not technique_list:
            raise ValueError("techniques must contain at least one entry")

        if self._rng.random() < self._epsilon:
            return self._rng.choice(technique_list)

        estimates = {t: self._estimate(context=context, technique=t) for t in technique_list}
        best = max(estimates.values())
        winners = [t for t, value in estimates.items() if value >= best - self._TIE_TOL]
        return self._rng.choice(winners)

    def record_outcome(self, *, context: str, technique: str, success: bool) -> None:
        """
        Record the outcome of an attack attempt for a given technique and context.

        Args:
            context (str): The context key the decision was made under.
            technique (str): The technique that was tried.
            success (bool): Whether the attempt succeeded.
        """
        successes, attempts = self._counts.get((context, technique), (0, 0))
        attempts += 1
        if success:
            successes += 1
        self._counts[(context, technique)] = (successes, attempts)

        global_successes, global_attempts = self._global_counts.get(technique, (0, 0))
        global_attempts += 1
        if success:
            global_successes += 1
        self._global_counts[technique] = (global_successes, global_attempts)

    def success_rate(self, *, context: str, technique: str) -> float:
        """
        Return the Laplace-smoothed success-rate estimate for a technique in a context.

        The "smoothed" rate is ``(successes + 1) / (attempts + 1)`` — Laplace smoothing
        provides an optimistic prior for unseen techniques (estimate = 1.0) and avoids
        division by zero. This is the same value used internally for exploitation decisions.
        """
        return self._estimate(context=context, technique=technique)

    def counts(self, *, context: str, technique: str) -> tuple[int, int]:
        """Return raw ``(successes, attempts)`` for a ``(context, technique)`` cell."""
        return self._counts.get((context, technique), (0, 0))

    def snapshot(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Return a shallow copy of the full counts table (for logging/debug)."""
        return dict(self._counts)

    def _estimate(self, *, context: str, technique: str) -> float:
        """
        Laplace-smoothed success-rate estimate for ``(context, technique)``.

        Below ``pool_threshold`` local attempts, the estimate uses the
        pooled-global success rate for the technique across all contexts.
        """
        local_s, local_n = self._counts.get((context, technique), (0, 0))
        if local_n >= self._pool_threshold:
            return (local_s + 1) / (local_n + 1)
        global_s, global_n = self._global_counts.get(technique, (0, 0))
        return (global_s + 1) / (global_n + 1)
