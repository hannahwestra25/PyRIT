# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Adaptive technique selection for the ``TextAdaptive`` scenario.

This module provides:
    - ``AdaptiveTechniqueSelector``: an epsilon-greedy bandit keyed by
      ``(context, technique)`` that tracks successes/attempts per arm and
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
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from pyrit.models.seeds.seed_attack_group import SeedAttackGroup


ContextExtractor = Callable[["SeedAttackGroup"], str]
"""Maps a ``SeedAttackGroup`` to a bandit context key."""


GLOBAL_CONTEXT: str = "_global"
UNCATEGORIZED_CONTEXT: str = "_uncategorized"


def global_context(_seed_attack_group: "SeedAttackGroup") -> str:
    """Return a constant context so all objectives share one bandit table."""
    return GLOBAL_CONTEXT


def harm_category_context(seed_attack_group: "SeedAttackGroup") -> str:
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
    and ``update`` records the outcome of an attempt.

    Selection uses epsilon-greedy with optimistic initialization:
        - With probability ``epsilon``, pick uniformly at random from ``arms``.
        - Otherwise, pick the arm with the highest estimated success rate.
          The estimate is ``(successes + 1) / (attempts + 1)``, so unseen
          arms look like 100% success and are explored first via tiebreak.

    When a ``(context, arm)`` cell has fewer than ``pool_threshold`` attempts,
    the estimate falls back to the pooled global rate for that arm across all
    contexts. This lets per-context bandits benefit from cross-context data
    until they have enough local samples. Set ``pool_threshold=1`` to disable
    pooling (use the local estimate as soon as any attempt is recorded).

    Note:
        This class is not thread/async safe. It assumes sequential calls,
        which matches the base ``Scenario._execute_scenario_async`` loop.
    """

    # Tolerance for tiebreaking in exploitation. Estimates are rational today,
    # so equality works, but this guards against future estimators that may
    # introduce floating-point drift.
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
            pool_threshold (int): Minimum per-(context, arm) attempts before
                the local estimate replaces the pooled-global estimate. Must
                be >= 1; set to 1 to disable pooling. Defaults to 3.
            rng (random.Random | None): Seedable RNG for deterministic tests.
                Defaults to a fresh ``random.Random()``.

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

    def select(self, *, context: str, arms: Sequence[str]) -> str:
        """
        Pick the next arm to try for ``context``.

        Args:
            context (str): The context key (e.g. ``"_global"`` or a harm category).
            arms (Sequence[str]): The candidate technique names.

        Returns:
            str: The chosen arm name.

        Raises:
            ValueError: If ``arms`` is empty.
        """
        arm_list = list(arms)
        if not arm_list:
            raise ValueError("arms must contain at least one technique")

        if self._rng.random() < self._epsilon:
            return self._rng.choice(arm_list)

        estimates = {arm: self._estimate(context=context, arm=arm) for arm in arm_list}
        best = max(estimates.values())
        winners = [arm for arm, value in estimates.items() if value >= best - self._TIE_TOL]
        return self._rng.choice(winners)

    def update(self, *, context: str, technique: str, success: bool) -> None:
        """
        Record the outcome of an attempt.

        Args:
            context (str): The context key the decision was made under.
            technique (str): The arm that was tried.
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
        Return the smoothed success-rate estimate for an arm in a context.

        This is the same value used internally for exploitation decisions.
        """
        return self._estimate(context=context, arm=technique)

    def counts(self, *, context: str, technique: str) -> tuple[int, int]:
        """Return raw ``(successes, attempts)`` for a ``(context, technique)`` cell."""
        return self._counts.get((context, technique), (0, 0))

    def snapshot(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Return a shallow copy of the full counts table (for logging/debug)."""
        return dict(self._counts)

    def _estimate(self, *, context: str, arm: str) -> float:
        """
        Smoothed success-rate estimate for ``(context, arm)``.

        Below ``pool_threshold`` local attempts, the estimate uses the
        pooled-global success rate for the arm across all contexts.
        """
        local_s, local_n = self._counts.get((context, arm), (0, 0))
        if local_n >= self._pool_threshold:
            return (local_s + 1) / (local_n + 1)
        global_s, global_n = self._global_counts.get(arm, (0, 0))
        return (global_s + 1) / (global_n + 1)
