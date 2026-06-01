# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.analytics.technique_analysis import (
    _DEFAULT_TECHNIQUE_LABEL_KEY,
    compute_technique_stats,
)
from pyrit.models import AttackOutcome

LABEL_KEY = _DEFAULT_TECHNIQUE_LABEL_KEY


def _make_result(*, technique: str, outcome: AttackOutcome) -> MagicMock:
    r = MagicMock()
    r.labels = {LABEL_KEY: technique}
    r.outcome = outcome
    return r


@pytest.fixture(autouse=True)
def _patch_memory():
    mock_memory = MagicMock()
    mock_memory.get_attack_results.return_value = []
    with patch("pyrit.analytics.technique_analysis.CentralMemory") as cm:
        cm.get_memory_instance.return_value = mock_memory
        yield mock_memory


class TestComputeTechniqueStats:
    def test_empty_results_returns_empty(self, _patch_memory):
        stats = compute_technique_stats(technique_eval_hashes=["a", "b"])
        assert stats == {}

    def test_counts_successes_and_failures(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
            _make_result(technique="b", outcome=AttackOutcome.FAILURE),
        ]

        stats = compute_technique_stats(technique_eval_hashes=["a", "b"])

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

        stats = compute_technique_stats(technique_eval_hashes=["a"])

        assert stats["a"].errors == 1
        assert stats["a"].undetermined == 1

    def test_ignores_techniques_not_in_requested_list(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="c", outcome=AttackOutcome.SUCCESS),
        ]

        stats = compute_technique_stats(technique_eval_hashes=["a", "b"])

        assert "a" in stats
        assert "c" not in stats

    def test_default_label_key_used_when_omitted(self, _patch_memory):
        compute_technique_stats(technique_eval_hashes=["x"])

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["labels"] == {LABEL_KEY: ["x"]}

    def test_passes_custom_label_key_to_memory_query(self, _patch_memory):
        custom_key = "my_custom_key"
        compute_technique_stats(technique_eval_hashes=["x"], label_key=custom_key)

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["labels"] == {custom_key: ["x"]}
        assert call_kwargs["scenario_result_id"] is None

    def test_passes_scenario_result_id_to_memory_query(self, _patch_memory):
        compute_technique_stats(technique_eval_hashes=["x"], scenario_result_id="run-123")

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["scenario_result_id"] == "run-123"

    def test_omits_techniques_with_no_history(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
        ]

        stats = compute_technique_stats(technique_eval_hashes=["a", "b"])

        assert "a" in stats
        assert "b" not in stats

    def test_success_rate_computed(self, _patch_memory):
        _patch_memory.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
            _make_result(technique="a", outcome=AttackOutcome.FAILURE),
        ]

        stats = compute_technique_stats(technique_eval_hashes=["a"])

        assert stats["a"].success_rate == pytest.approx(0.5)

    def test_passes_attack_classes_to_memory_query(self, _patch_memory):
        compute_technique_stats(
            technique_eval_hashes=["x"],
            attack_classes=["TextAdaptive", "ImageAdaptive"],
        )

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["attack_classes"] == ["TextAdaptive", "ImageAdaptive"]

    def test_passes_harm_categories_to_memory_query(self, _patch_memory):
        compute_technique_stats(
            technique_eval_hashes=["x"],
            targeted_harm_categories=["misinformation", "hate"],
        )

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["targeted_harm_categories"] == ["misinformation", "hate"]

    def test_merges_extra_labels_with_technique_label(self, _patch_memory):
        compute_technique_stats(
            technique_eval_hashes=["x"],
            extra_labels={"experiment": "ablation_v3", "tier": ["a", "b"]},
        )

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["labels"] == {
            LABEL_KEY: ["x"],
            "experiment": "ablation_v3",
            "tier": ["a", "b"],
        }

    def test_extra_labels_cannot_override_technique_label(self, _patch_memory):
        compute_technique_stats(
            technique_eval_hashes=["x"],
            extra_labels={LABEL_KEY: ["evil"], "other": "ok"},
        )

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["labels"] == {LABEL_KEY: ["x"], "other": "ok"}

    def test_default_filter_kwargs_are_none(self, _patch_memory):
        compute_technique_stats(technique_eval_hashes=["x"])

        call_kwargs = _patch_memory.get_attack_results.call_args[1]
        assert call_kwargs["attack_classes"] is None
        assert call_kwargs["targeted_harm_categories"] is None

    def test_injected_memory_bypasses_central_memory(self, _patch_memory):
        injected = MagicMock()
        injected.get_attack_results.return_value = [
            _make_result(technique="a", outcome=AttackOutcome.SUCCESS),
        ]

        stats = compute_technique_stats(technique_eval_hashes=["a"], memory=injected)

        injected.get_attack_results.assert_called_once()
        _patch_memory.get_attack_results.assert_not_called()
        assert stats["a"].successes == 1

    def test_default_label_key_matches_adaptive_constant(self):
        from pyrit.scenario.scenarios.adaptive.selectors.technique_selector import (
            ADAPTIVE_TECHNIQUE_LABEL,
        )

        assert _DEFAULT_TECHNIQUE_LABEL_KEY == ADAPTIVE_TECHNIQUE_LABEL
