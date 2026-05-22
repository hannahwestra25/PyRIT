# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Epsilon-greedy technique selector for adaptive scenarios."""

from __future__ import annotations

import hashlib
import logging
import random
import struct
from collections.abc import Sequence

from pyrit.analytics.result_analysis import AttackStats
from pyrit.analytics.scenario_analysis import compute_technique_success_rates
from pyrit.scenario.scenarios.adaptive.selectors.technique_selector import ADAPTIVE_TECHNIQUE_LABEL, SelectorScope

logger = logging.getLogger(__name__)


def _derive_rng(random_seed: int | None, decision_key: str) -> random.Random:
    """
    Derive a per-decision ``Random`` from ``(random_seed, decision_key)``.

    Returns:
        random.Random: A fresh ``random.Random`` seeded deterministically from the
        inputs when ``random_seed`` is not None, or an unseeded ``Random`` otherwise.
    """
    if random_seed is None:
        return random.Random()
    digest = hashlib.sha256(f"{random_seed}|{decision_key}".encode()).digest()
    derived_seed = struct.unpack("<Q", digest[:8])[0]
    return random.Random(derived_seed)


class EpsilonGreedyTechniqueSelector:
    """
    Stateless epsilon-greedy selector over attack techniques.

    Queries memory for historical success rates and applies epsilon-greedy
    selection. With probability ``epsilon`` picks uniformly at random;
    otherwise picks the technique with the highest Laplace-smoothed estimate
    ``(s + 1) / (n + 1)`` (unseen techniques start at 1.0).

    The selector is **stateless** — it does not maintain internal counts.
    All outcome data comes from the memory database via
    ``_compute_success_rates``. Calling ``select_async`` with the same
    arguments produces the same result (deterministic given memory
    contents and ``random_seed``).
    """

    _TIE_TOL: float = 1e-12

    def __init__(
        self,
        *,
        epsilon: float = 0.2,
        scope: SelectorScope = SelectorScope.ALL_RUNS,
        random_seed: int | None = None,
    ) -> None:
        """
        Args:
            epsilon (float): Exploration probability in [0.0, 1.0]. Defaults to 0.2.
            scope (SelectorScope): Whether to use all historical data or only
                the current scenario run. Defaults to ``SelectorScope.ALL_RUNS``.
            random_seed (int | None): Base seed for deterministic per-decision RNG
                derivation. Defaults to ``None`` (non-deterministic).

        Raises:
            ValueError: If ``epsilon`` is outside [0.0, 1.0].
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")

        self._epsilon = epsilon
        self._scope = scope
        self._seed = random_seed

    async def select_async(
        self,
        *,
        technique_identifiers: Sequence[str],
        objective: str,
        num_top_techniques: int = 1,
        scenario_result_id: str | None = None,
    ) -> Sequence[str]:
        """
        Return up to ``num_top_techniques`` techniques in priority order.

        Args:
            technique_identifiers (Sequence[str]): Available technique names.
            objective (str): The objective text for scoping the per-decision RNG.
            num_top_techniques (int): Max techniques to return. Defaults to 1.
            scenario_result_id (str | None): If provided, restrict memory
                queries to this scenario run. Defaults to ``None`` (all runs).

        Returns:
            Sequence[str]: Techniques in priority order. Fewer than
                ``num_top_techniques`` if not enough techniques are available.

        Raises:
            ValueError: If ``technique_identifiers`` is empty.
        """
        technique_list = list(technique_identifiers)
        if not technique_list:
            raise ValueError("technique_identifiers must contain at least one entry")

        num_top_techniques = min(num_top_techniques, len(technique_list))

        decision_key = objective
        rng = _derive_rng(self._seed, decision_key)

        stats = compute_technique_success_rates(
            technique_hashes=technique_list,
            label_key=ADAPTIVE_TECHNIQUE_LABEL,
            scenario_result_id=scenario_result_id if self._scope == SelectorScope.CURRENT_RUN else None,
        )

        chosen: list[str] = []
        remaining = list(technique_list)

        for _ in range(num_top_techniques):
            if not remaining:
                break

            if rng.random() < self._epsilon:
                pick = rng.choice(remaining)
            else:
                estimates = {
                    t: self._estimate(technique=t, stats=stats) for t in remaining
                }
                best = max(estimates.values())
                winners = [t for t, v in estimates.items() if v >= best - self._TIE_TOL]
                pick = rng.choice(winners)

            chosen.append(pick)
            remaining.remove(pick)

        return chosen

    @staticmethod
    def _estimate(*, technique: str, stats: dict[str, AttackStats]) -> float:
        """
        Laplace-smoothed success-rate estimate for a technique.

        Unseen techniques get ``(0 + 1) / (0 + 1) = 1.0`` (optimistic init).

        Args:
            technique (str): The technique name.
            stats (dict[str, AttackStats]): Pre-computed stats from memory.

        Returns:
            float: Estimated success rate in ``(0, 1]``.
        """
        technique_stats = stats.get(technique)
        if technique_stats is None or technique_stats.total_decided == 0:
            return 1.0
        return (technique_stats.successes + 1) / (technique_stats.total_decided + 1)
