# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from unittest.mock import MagicMock

import pytest

from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    UNCATEGORIZED_CONTEXT,
    AdaptiveTechniqueSelector,
    global_context,
    harm_category_context,
)

ARMS = ["a", "b", "c", "d"]


def _seeded_selector(*, epsilon: float = 0.0, pool_threshold: int = 3, seed: int = 0) -> AdaptiveTechniqueSelector:
    return AdaptiveTechniqueSelector(
        epsilon=epsilon,
        pool_threshold=pool_threshold,
        rng=random.Random(seed),
    )


class TestAdaptiveTechniqueSelectorInit:
    def test_init_defaults(self):
        selector = AdaptiveTechniqueSelector()
        assert selector.snapshot() == {}

    @pytest.mark.parametrize("bad_epsilon", [-0.1, 1.1, 2.0, -1.0])
    def test_init_rejects_out_of_range_epsilon(self, bad_epsilon):
        with pytest.raises(ValueError, match="epsilon"):
            AdaptiveTechniqueSelector(epsilon=bad_epsilon)

    def test_init_rejects_pool_threshold_below_one(self):
        with pytest.raises(ValueError, match="pool_threshold"):
            AdaptiveTechniqueSelector(pool_threshold=0)
        with pytest.raises(ValueError, match="pool_threshold"):
            AdaptiveTechniqueSelector(pool_threshold=-1)


class TestAdaptiveTechniqueSelectorSelect:
    def test_select_empty_arms_raises(self):
        selector = _seeded_selector()
        with pytest.raises(ValueError, match="arms"):
            selector.select(context=GLOBAL_CONTEXT, arms=[])

    def test_select_all_unseen_ties_resolved_randomly(self):
        # With epsilon=0 and an empty table, every arm has estimate 1/1=1.0,
        # so the result is the seeded random tiebreak. Different seeds should
        # be able to produce different winners.
        winners = {_seeded_selector(seed=s).select(context=GLOBAL_CONTEXT, arms=ARMS) for s in range(50)}
        assert len(winners) > 1
        assert winners.issubset(set(ARMS))

    def test_select_exploits_clear_winner(self):
        selector = _seeded_selector(pool_threshold=1)
        # Give "b" a track record of pure success, others pure failure.
        for _ in range(5):
            selector.update(context=GLOBAL_CONTEXT, technique="b", success=True)
        for arm in ("a", "c", "d"):
            for _ in range(5):
                selector.update(context=GLOBAL_CONTEXT, technique=arm, success=False)

        # With epsilon=0, every selection must exploit the winner.
        for _ in range(20):
            assert selector.select(context=GLOBAL_CONTEXT, arms=ARMS) == "b"

    def test_select_epsilon_one_is_pure_random(self):
        selector = _seeded_selector(epsilon=1.0)
        # Bias the table heavily toward "a"; with epsilon=1 it must still be ignored.
        for _ in range(20):
            selector.update(context=GLOBAL_CONTEXT, technique="a", success=True)

        picks = [selector.select(context=GLOBAL_CONTEXT, arms=ARMS) for _ in range(200)]
        assert set(picks) == set(ARMS)

    def test_select_epsilon_zero_never_explores(self):
        selector = _seeded_selector(epsilon=0.0, pool_threshold=1)
        for _ in range(3):
            selector.update(context=GLOBAL_CONTEXT, technique="a", success=True)
        # Make the other arms tried-and-failed so they fall below "a"'s estimate;
        # unseen arms would otherwise tie at the optimistic 1.0.
        for arm in ("b", "c", "d"):
            selector.update(context=GLOBAL_CONTEXT, technique=arm, success=False)
        for _ in range(50):
            assert selector.select(context=GLOBAL_CONTEXT, arms=ARMS) == "a"

    def test_select_cold_start_round_robins(self):
        # Optimistic init + epsilon=0: untried arms tie at 1.0 and beat tried-and-failed
        # arms (1/2 = 0.5). So the first failures push each arm to "tried" exactly once
        # before any arm gets tried twice.
        selector = _seeded_selector(pool_threshold=1)
        tried: list[str] = []
        for _ in range(len(ARMS)):
            arm = selector.select(context=GLOBAL_CONTEXT, arms=ARMS)
            tried.append(arm)
            selector.update(context=GLOBAL_CONTEXT, technique=arm, success=False)
        assert sorted(tried) == sorted(ARMS)


class TestAdaptiveTechniqueSelectorUpdate:
    def test_update_accumulates_counts(self):
        selector = _seeded_selector()
        selector.update(context="ctx", technique="a", success=True)
        selector.update(context="ctx", technique="a", success=False)
        selector.update(context="ctx", technique="a", success=True)
        assert selector.counts(context="ctx", technique="a") == (2, 3)

    def test_update_separate_contexts_are_independent(self):
        selector = _seeded_selector()
        selector.update(context="x", technique="a", success=True)
        selector.update(context="y", technique="a", success=False)
        assert selector.counts(context="x", technique="a") == (1, 1)
        assert selector.counts(context="y", technique="a") == (0, 1)

    def test_counts_default_zero_for_unseen(self):
        selector = _seeded_selector()
        assert selector.counts(context="missing", technique="missing") == (0, 0)

    def test_update_keeps_pooled_global_counts_in_sync(self):
        # Pooled-global counts back the O(1) pooled-backoff branch in _estimate.
        # They must aggregate across contexts for a given arm.
        selector = _seeded_selector(pool_threshold=5)
        selector.update(context="x", technique="a", success=True)
        selector.update(context="y", technique="a", success=False)
        selector.update(context="z", technique="a", success=True)
        selector.update(context="x", technique="b", success=True)

        # Below the local threshold, _estimate must use the pooled-global rate.
        # arm "a": 2 successes / 3 attempts -> (2+1)/(3+1) = 0.75
        assert selector.success_rate(context="new_ctx", technique="a") == pytest.approx(0.75)
        # arm "b": 1/1 -> (1+1)/(1+1) = 1.0
        assert selector.success_rate(context="new_ctx", technique="b") == pytest.approx(1.0)
        # Unseen arm "c" -> (0+1)/(0+1) = 1.0
        assert selector.success_rate(context="new_ctx", technique="c") == pytest.approx(1.0)


class TestAdaptiveTechniqueSelectorEstimate:
    def test_success_rate_unseen_is_one(self):
        # Optimistic init: (0 + 1) / (0 + 1) = 1.0
        selector = _seeded_selector()
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(1.0)

    def test_success_rate_local_when_above_threshold(self):
        selector = _seeded_selector(pool_threshold=2)
        for _ in range(3):
            selector.update(context="ctx", technique="a", success=True)
        # (3 + 1) / (3 + 1) = 1.0
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(1.0)

    def test_success_rate_pools_when_below_threshold(self):
        selector = _seeded_selector(pool_threshold=5)
        # Local cell has only 1 attempt (below threshold).
        selector.update(context="ctx", technique="a", success=False)
        # Other contexts have plenty of data for arm "a".
        for _ in range(10):
            selector.update(context="other", technique="a", success=True)
        # Pooled estimate = (10 + 0 + 1) / (10 + 1 + 1) = 11/12.
        assert selector.success_rate(context="ctx", technique="a") == pytest.approx(11 / 12)


class TestContextExtractors:
    def test_global_context_is_constant(self):
        sg = MagicMock()
        assert global_context(sg) == GLOBAL_CONTEXT

    def test_harm_category_context_uses_first_category(self):
        sg = MagicMock()
        sg.harm_categories = ["violence", "hate"]
        assert harm_category_context(sg) == "violence"

    def test_harm_category_context_falls_back_when_empty(self):
        sg = MagicMock()
        sg.harm_categories = []
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT

    def test_harm_category_context_falls_back_when_none(self):
        sg = MagicMock()
        sg.harm_categories = None
        assert harm_category_context(sg) == UNCATEGORIZED_CONTEXT
