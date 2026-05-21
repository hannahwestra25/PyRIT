# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Epsilon-greedy technique selector for adaptive scenarios."""

from __future__ import annotations

import hashlib
import random
import struct
import threading
from collections.abc import Sequence


def _derive_rng(random_seed: int | None, context: str, decision_key: str) -> random.Random:
    """
    Derive a per-decision ``Random`` from ``(random_seed, context, decision_key)``.

    Returns a fresh ``random.Random`` seeded deterministically from the
    inputs when ``random_seed`` is not None, or an unseeded ``Random`` otherwise.
    """
    if random_seed is None:
        return random.Random()
    digest = hashlib.sha256(f"{random_seed}|{context}|{decision_key}".encode()).digest()
    derived_seed = struct.unpack("<Q", digest[:8])[0]
    return random.Random(derived_seed)


class EpsilonGreedyTechniqueSelector:
    """
    Epsilon-greedy selector over attack techniques.

    Maintains a ``(context, technique) -> (successes, attempts)`` table. With
    probability ``epsilon`` picks uniformly at random; otherwise picks the
    technique with the highest Laplace-smoothed estimate ``(s + 1) / (n + 1)``
    (unseen techniques start at 1.0). A ``(context, technique)`` cell with
    fewer than ``pool_threshold`` attempts falls back to the technique's
    pooled rate across all contexts.

    Each ``select`` call derives a per-decision ``Random`` from
    ``(random_seed, context, decision_key)`` so that resume produces deterministic
    decisions without persisting RNG state.

    All public methods are guarded by a ``threading.Lock`` so concurrent
    callers cannot corrupt the table. The lock makes individual ops atomic,
    not the overall select → execute → record sequence.
    """

    # Tolerance for tiebreaking on float estimates (current estimates are exact
    # rationals; this guards against future estimator changes).
    _TIE_TOL: float = 1e-12

    def __init__(
        self,
        *,
        epsilon: float = 0.2,
        pool_threshold: int = 3,
        random_seed: int | None = None,
    ) -> None:
        """
        Args:
            epsilon (float): Exploration probability in [0.0, 1.0]. Defaults to 0.2.
            pool_threshold (int): Minimum per-(context, technique) attempts before
                the local estimate replaces the pooled rate. Must be >= 1; set to 1
                to disable pooling. Defaults to 3.
            random_seed (int | None): Base seed for deterministic per-decision RNG derivation.
                Defaults to ``None`` (non-deterministic).

        Raises:
            ValueError: If ``epsilon`` is outside [0.0, 1.0] or ``pool_threshold`` < 1.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        if pool_threshold < 1:
            raise ValueError(f"pool_threshold must be >= 1, got {pool_threshold}")

        self._epsilon = epsilon
        self._pool_threshold = pool_threshold
        self._seed = random_seed
        self._counts: dict[tuple[str, str], tuple[int, int]] = {}
        # Per-technique pooled counts, kept in sync with ``_counts`` so the
        # pooled-backoff branch in ``_estimate`` is O(1).
        self._global_counts: dict[str, tuple[int, int]] = {}
        # Monotonic counter for auto-generating decision keys when the caller
        # doesn't provide one.
        self._decision_counter: int = 0
        # Guards _counts, _global_counts, and _decision_counter against concurrent callers.
        self._lock = threading.Lock()

    def select(self, *, context: str, techniques: Sequence[str], decision_key: str = "") -> str:
        """
        Pick the next technique to try for ``context``.

        Args:
            context (str): The context key.
            techniques (Sequence[str]): Candidate technique names.
            decision_key (str): Caller-supplied key (e.g. ``"obj_id:attempt_idx"``)
                used to derive a per-decision RNG for deterministic replay.
                Defaults to ``""`` (auto-incremented counter).

        Returns:
            str: The chosen technique name.

        Raises:
            ValueError: If ``techniques`` is empty.
        """
        technique_list = list(techniques)
        if not technique_list:
            raise ValueError("techniques must contain at least one entry")

        with self._lock:
            if decision_key:
                effective_key = decision_key
            else:
                effective_key = str(self._decision_counter)
                self._decision_counter += 1
            rng = _derive_rng(self._seed, context, effective_key)

            if rng.random() < self._epsilon:
                return rng.choice(technique_list)

            estimates = {t: self._estimate(context=context, technique=t) for t in technique_list}
            best = max(estimates.values())
            winners = [t for t, value in estimates.items() if value >= best - self._TIE_TOL]
            return rng.choice(winners)

    def record_outcome(self, *, context: str, technique: str, success: bool) -> None:
        """
        Record the outcome of an attempt.

        Args:
            context (str): The context key the decision was made under.
            technique (str): The technique that was tried.
            success (bool): Whether the attempt succeeded.
        """
        with self._lock:
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
        """Return the Laplace-smoothed estimate ``(s + 1) / (n + 1)`` used for exploitation."""
        with self._lock:
            return self._estimate(context=context, technique=technique)

    def counts(self, *, context: str, technique: str) -> tuple[int, int]:
        """Return raw ``(successes, attempts)`` for a ``(context, technique)`` cell."""
        with self._lock:
            return self._counts.get((context, technique), (0, 0))

    def snapshot(self) -> dict[tuple[str, str], tuple[int, int]]:
        """Return a shallow copy of the full counts table (for logging/debug)."""
        with self._lock:
            return dict(self._counts)

    def _estimate(self, *, context: str, technique: str) -> float:
        """
        Estimate for ``(context, technique)``; falls back to pooled rate below
        ``pool_threshold`` local attempts.

        Callers must already hold ``self._lock``.

        Returns:
            float: Laplace-smoothed success-rate estimate in ``(0, 1)``.
        """
        local_s, local_n = self._counts.get((context, technique), (0, 0))
        if local_n >= self._pool_threshold:
            return (local_s + 1) / (local_n + 1)
        global_s, global_n = self._global_counts.get(technique, (0, 0))
        return (global_s + 1) / (global_n + 1)

