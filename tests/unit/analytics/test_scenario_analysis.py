# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.analytics.scenario_analysis import compute_technique_success_rates
from pyrit.models import AttackOutcome


LABEL_KEY = "_adaptive_technique"


def _make_result(*, technique: str, outcome: AttackOutcome) -> MagicMock:
    r = MagicMock()
    r.labels = {LABEL_KEY: technique}
    r.outcome = outcome
    return r


@pytest.fixture(autouse=True)
def _patch_memory():
    mock_memory = MagicMock()
    mock_memory.get_attack_results.return_value = []
    with patch("pyrit.analytics.scenario_analysis.CentralMemory") as cm:
        cm.get_memory_instance.return_value = mock_memory
        yield mock_memory


class TestComputeTechniqueSuccessRates:

    def test_empty_results_returns_empty(self, _patch_memory):
        stats = compute_technique_success_rates(technique_hashes=["a", "b"], label_key=LABEL_KEY)
        assert stats == {}

    def test_counts_successes_and_failures(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
            _make_result(technique="b", outcome=AttackOutcome.FAILURE),
        ]

        stats = compute_technique_success_rates(technique_hashes=["a", "b"], label_key=LABEL_KEY)

        assert stats["a"].successes == 2
        assert stats["a"].failures == 1
        assert stats["a"].total_decided == 3
        assert stats["b"].successes == 0
        assert stats["b"].failures == 1

    def test_counts_errors_and_undetermined(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.ERROR),
            _make_result(technique="a", outcome=AttackOutcome.UNDETERMINED),
        ]

        stats = compute_technique_success_rates(technique_hashes=["a"], label_key=LABEL_KEY)

        assert stats["a"].errors == 1
        assert stats["a"].undetermined == 1

    def test_ignores_techniques_not_in_requested_list(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="c", outcome=AttackOutcome.SUCCESS),
        ]

        stats = compute_technique_success_rates(technique_hashes=["a", "b"], label_key=LABEL_KEY)

        assert "a" in stats
        assert "c" not in stats

    def test_passes_label_key_to_memory_query(self, _patch_memory):
        custom_key = "my_custom_key"
        compute_technique_success_rates(technique_hashes=["x"], label_key=custom_key)

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["labels"] == {custom_key: ["x"]}
        assert call_kwargs["scenario_result_id"] is None

    def test_passes_scenario_result_id_to_memory_query(self, _patch_memory):
        compute_technique_success_rates(
            technique_hashes=["x"], label_key=LABEL_KEY, scenario_result_id="run-123"
        )

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["scenario_result_id"] == "run-123"

    def test_omits_techniques_with_no_history(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
        ]

        stats = compute_technique_success_rates(technique_hashes=["a", "b"], label_key=LABEL_KEY)

        assert "a" in stats
        assert "b" not in stats

    def test_success_rate_computed(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
        ]

        stats = compute_technique_success_rates(technique_hashes=["a"], label_key=LABEL_KEY)

        assert stats["a"].success_rate == pytest.approx(0.5)
