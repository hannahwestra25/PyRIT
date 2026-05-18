# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_CONTEXT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    AdaptiveDispatchAttack,
    AdaptiveDispatchContext,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)


def _make_inner_attack(*, name: str, outcomes: list[AttackOutcome]) -> MagicMock:
    """Build a mocked inner attack whose execute_async returns the given outcomes in order."""
    inner = MagicMock(name=name)
    results = [
        AttackResult(
            conversation_id=f"conv-{name}-{i}",
            objective="obj",
            outcome=outcome,
        )
        for i, outcome in enumerate(outcomes)
    ]
    inner.execute_async = AsyncMock(side_effect=results)
    return inner


def _make_context(*, objective: str = "obj", labels: dict[str, str] | None = None) -> AdaptiveDispatchContext:
    return AdaptiveDispatchContext(params=AttackParameters(objective=objective, memory_labels=labels or {}))


@pytest.fixture
def selector() -> AdaptiveTechniqueSelector:
    # epsilon=0 makes selection deterministic given the table.
    return AdaptiveTechniqueSelector(epsilon=0.0, pool_threshold=1, rng=random.Random(0))


@pytest.fixture
def target() -> MagicMock:
    return MagicMock(name="objective_target")


class TestInit:
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_empty_techniques(self, target, selector):
        with pytest.raises(ValueError, match="techniques"):
            AdaptiveDispatchAttack(objective_target=target, techniques={}, selector=selector)

    @pytest.mark.parametrize("bad_max", [0, -1])
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_invalid_max_attempts(self, target, selector, bad_max):
        with pytest.raises(ValueError, match="max_attempts_per_objective"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={"a": _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])},
                selector=selector,
                max_attempts_per_objective=bad_max,
            )


@pytest.mark.usefixtures("patch_central_database")
class TestPerform:
    async def test_stops_on_first_success(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])
        b = _make_inner_attack(name="b", outcomes=[AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a, "b": b},
            selector=selector,
            max_attempts_per_objective=5,
        )

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.SUCCESS
        total_calls = a.execute_async.call_count + b.execute_async.call_count
        assert total_calls == 1

    async def test_retries_until_max_attempts_on_failure(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.FAILURE] * 3)
        b = _make_inner_attack(name="b", outcomes=[AttackOutcome.FAILURE] * 3)
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a, "b": b},
            selector=selector,
            max_attempts_per_objective=3,
        )

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.FAILURE
        total_calls = a.execute_async.call_count + b.execute_async.call_count
        assert total_calls == 3

    async def test_updates_selector_on_each_attempt(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS])
        b = _make_inner_attack(name="b", outcomes=[AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a, "b": b},
            selector=selector,
            max_attempts_per_objective=3,
        )

        await dispatcher._perform_async(context=_make_context())

        # Total attempts across arms must equal sum of selector counts.
        total_attempts = sum(selector.counts(context=GLOBAL_CONTEXT, technique=t)[1] for t in ("a", "b"))
        total_calls = a.execute_async.call_count + b.execute_async.call_count
        assert total_attempts == total_calls

    async def test_passes_objective_to_inner(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a},
            selector=selector,
        )

        await dispatcher._perform_async(context=_make_context(objective="my-goal"))

        kwargs = a.execute_async.call_args.kwargs
        assert kwargs["objective"] == "my-goal"

    async def test_attaches_technique_and_attempt_labels(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a},
            selector=selector,
        )

        await dispatcher._perform_async(context=_make_context(labels={"foo": "bar"}))

        labels = a.execute_async.call_args.kwargs["memory_labels"]
        assert labels["foo"] == "bar"  # caller labels preserved
        assert labels[ADAPTIVE_TECHNIQUE_LABEL] == "a"
        assert labels[ADAPTIVE_ATTEMPT_LABEL] == "1"

    async def test_uses_adaptive_context_from_label(self, target, selector):
        # Two techniques; one has been heavily rewarded under context "violence" only.
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])
        b = _make_inner_attack(name="b", outcomes=[AttackOutcome.SUCCESS])
        for _ in range(5):
            selector.record_outcome(context="violence", technique="b", success=True)
        for _ in range(5):
            selector.record_outcome(context="violence", technique="a", success=False)

        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a, "b": b},
            selector=selector,
        )
        ctx = _make_context(labels={ADAPTIVE_CONTEXT_LABEL: "violence"})
        await dispatcher._perform_async(context=ctx)

        # Exploit should have picked "b" first.
        assert b.execute_async.call_count == 1
        assert a.execute_async.call_count == 0

    async def test_falls_back_to_global_context_when_label_missing(self, target, selector):
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a},
            selector=selector,
        )
        await dispatcher._perform_async(context=_make_context(labels={}))

        # The global context bucket received the update.
        assert selector.counts(context=GLOBAL_CONTEXT, technique="a") == (1, 1)

    async def test_metadata_records_adaptive_trail(self, target, selector):
        # Technique "a" fails on the first attempt then succeeds; verify the trail
        # captures both attempts in order.
        a = _make_inner_attack(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": a},
            selector=selector,
            max_attempts_per_objective=3,
        )
        result = await dispatcher._perform_async(context=_make_context())

        trail = result.metadata["adaptive_attempts"]
        assert trail == [
            {"technique": "a", "outcome": "failure"},
            {"technique": "a", "outcome": "success"},
        ]
        assert result.metadata["adaptive_context"] == GLOBAL_CONTEXT


@pytest.mark.usefixtures("patch_central_database")
class TestValidate:
    @pytest.mark.parametrize("bad_objective", ["", "   ", "\n\t"])
    def test_validate_rejects_empty_objective(self, target, selector, bad_objective):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
        )
        with pytest.raises(ValueError, match="objective"):
            dispatcher._validate_context(context=_make_context(objective=bad_objective))

    def test_validate_accepts_normal_objective(self, target, selector):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_inner_attack(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
        )
        # Does not raise.
        dispatcher._validate_context(context=_make_context(objective="ok"))
