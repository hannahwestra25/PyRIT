# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scenario-level analytics: technique success rates and related helpers."""

from __future__ import annotations

from collections.abc import Sequence

from pyrit.analytics.result_analysis import AttackStats, _compute_stats
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome


def compute_technique_success_rates(
    *,
    technique_hashes: Sequence[str],
    label_key: str,
    scenario_result_id: str | None = None,
) -> dict[str, AttackStats]:
    """
    Query memory for historical success rates grouped by technique eval hash.

    Fetches all ``AttackResult`` rows whose memory labels contain
    ``label_key`` matching one of ``technique_hashes``, then aggregates
    outcomes into per-technique :class:`AttackStats`.

    By default queries across all scenario runs. Pass ``scenario_result_id``
    to restrict to a single run.

    Args:
        technique_hashes (Sequence[str]): Technique eval hashes to query.
        label_key (str): Memory-label key that stores the technique hash.
        scenario_result_id (str | None): If provided, restrict results to
            a single scenario run. Defaults to ``None`` (all runs).

    Returns:
        dict[str, AttackStats]: Stats per technique hash. Techniques with
            no history are omitted from the result.
    """

    memory = CentralMemory.get_memory_instance()
    results = memory.get_attack_results(
        labels={label_key: list(technique_hashes)},
        scenario_result_id=scenario_result_id,
    )

    counts: dict[str, tuple[int, int, int, int]] = {}
    for result in results:
        technique = result.labels.get(label_key)
        if not technique or technique not in technique_hashes:
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
