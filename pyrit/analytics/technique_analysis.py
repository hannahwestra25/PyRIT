# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scenario-level analytics: technique success rates and related helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyrit.analytics.result_analysis import AttackStats, _compute_stats
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyrit.memory.memory_interface import MemoryInterface


# Must match ``ADAPTIVE_TECHNIQUE_LABEL`` in
# ``pyrit.scenario.scenarios.adaptive.selectors.technique_selector``. Kept inline
# so the analytics layer doesn't import from the scenarios layer; a unit test
# asserts the two stay in sync.
_DEFAULT_TECHNIQUE_LABEL_KEY = "_adaptive_technique"


def compute_technique_stats(
    *,
    technique_eval_hashes: Sequence[str],
    label_key: str = _DEFAULT_TECHNIQUE_LABEL_KEY,
    scenario_result_id: str | None = None,
    attack_classes: Sequence[str] | None = None,
    targeted_harm_categories: Sequence[str] | None = None,
    extra_labels: Mapping[str, str | Sequence[str]] | None = None,
    memory: MemoryInterface | None = None,
) -> dict[str, AttackStats]:
    """
    Query memory for historical outcome stats grouped by technique eval hash.

    Fetches all ``AttackResult`` rows whose memory labels contain
    ``label_key`` matching one of ``technique_eval_hashes``, then aggregates
    outcomes into per-technique ``AttackStats``.

    By default queries across all scenario runs. Pass any subset of the
    optional filters to narrow the historical window.

    Args:
        technique_eval_hashes (Sequence[str]): Technique eval hashes to
            aggregate. Returned dict is keyed by these.
        label_key (str): Memory-label key that stores the technique hash.
            Defaults to the key the adaptive dispatcher stamps
            (``"_adaptive_technique"``); override only for custom callers.
        scenario_result_id (str | None): Restrict to a single scenario run.
            Defaults to ``None`` (aggregate across all runs).
        attack_classes (Sequence[str] | None): Restrict to results emitted
            by these attack / scenario class names. Forwarded to
            ``memory.get_attack_results``. Defaults to ``None``.
        targeted_harm_categories (Sequence[str] | None): Restrict to results
            whose prompts target these harm categories. Defaults to ``None``.
        extra_labels (Mapping[str, str | Sequence[str]] | None): Additional
            memory-label filters merged on top of the
            ``{label_key: technique_eval_hashes}`` filter the function always
            applies. Keys that collide with ``label_key`` are ignored
            (the technique-hash filter wins). Defaults to ``None``.
        memory (MemoryInterface | None): Memory backend to query. Defaults to
            ``CentralMemory.get_memory_instance()``.

    Returns:
        dict[str, AttackStats]: Stats per technique eval hash. Techniques
            with no matching history are omitted from the result.
    """
    labels: dict[str, str | Sequence[str]] = {label_key: list(technique_eval_hashes)}
    if extra_labels:
        for key, value in extra_labels.items():
            if key == label_key:
                continue
            labels[key] = value

    if memory is None:
        memory = CentralMemory.get_memory_instance()
    results = memory.get_attack_results(
        labels=labels,
        scenario_result_id=scenario_result_id,
        attack_classes=attack_classes,
        targeted_harm_categories=targeted_harm_categories,
    )

    counts: dict[str, tuple[int, int, int, int]] = {}
    for result in results:
        technique = result.labels.get(label_key)
        if not technique or technique not in technique_eval_hashes:
            continue

        s, f, u, e = counts.get(technique, (0, 0, 0, 0))
        if result.outcome == AttackOutcome.SUCCESS:
            counts[technique] = (s + 1, f, u, e)
        elif result.outcome == AttackOutcome.FAILURE:
            counts[technique] = (s, f + 1, u, e)
        elif result.outcome == AttackOutcome.ERROR:
            counts[technique] = (s, f, u, e + 1)
        else:
            counts[technique] = (s, f, u + 1, e)

    stats: dict[str, AttackStats] = {}
    for technique, (s, f, u, e) in counts.items():
        stats[technique] = _compute_stats(successes=s, failures=f, undetermined=u, errors=e)
    return stats
