# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import pytest

from pyrit.executor.attack.compound.sequential_attack import (
    SequentialAttack,
    SequentialAttackResult,
    SequentialChildAttack,
)
from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup, SeedObjective
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    AdaptiveDispatchAttack,
    AdaptiveDispatchContext,
    AdaptiveDispatchParams,
    TechniqueBundle,
)


def _make_bundle(*, name: str, outcomes: list[AttackOutcome], seed_technique=None) -> TechniqueBundle:
    """Build a TechniqueBundle whose attack stub yields the given outcomes in order."""
    attack = MagicMock(name=f"attack-{name}")
    attack._outcomes = outcomes
    attack._name = name
    return TechniqueBundle(attack=attack, name=name, seed_technique=seed_technique)


def _make_context(
    *,
    objective: str = "obj",
    labels: dict[str, str] | None = None,
    seed_group: SeedAttackGroup | None = None,
    harm_categories: list[str] | None = None,
) -> AdaptiveDispatchContext:
    if seed_group is None:
        seed_group = SeedAttackGroup(seeds=[SeedObjective(value=objective, harm_categories=harm_categories)])
    return AdaptiveDispatchContext(
        params=AdaptiveDispatchParams(
            objective=objective,
            memory_labels=labels or {},
            seed_group=seed_group,
        )
    )


def _patch_child_attack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundles: dict[str, TechniqueBundle],
) -> list[dict]:
    """
    Replace ``SequentialAttack._run_child_attack_async`` with a stub backed
    by per-bundle outcomes.

    Each invocation records the merged ``memory_labels`` and the resulting
    ``AttackResult`` so tests can inspect per-attempt routing and per-attempt
    label stamping without monkey-patching ``AttackExecutor``.
    """
    name_for_attack = {id(b.attack): name for name, b in bundles.items()}
    counters: dict[str, int] = dict.fromkeys(bundles, 0)
    calls: list[dict] = []

    async def _stub(self, *, child_attack: SequentialChildAttack, memory_labels: dict[str, str], attribution=None):
        name = name_for_attack[id(child_attack.strategy)]
        idx = counters[name]
        counters[name] = idx + 1
        outcome = child_attack.strategy._outcomes[idx]
        result = AttackResult(
            conversation_id=f"conv-{name}-{idx}",
            objective="obj",
            outcome=outcome,
        )
        calls.append(
            {
                "name": name,
                "attempt_labels": dict(memory_labels),
                "child_attack": child_attack,
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(SequentialAttack, "_run_child_attack_async", _stub, raising=True)
    return calls


class _StubSelector:
    """A deterministic selector stub that returns techniques in the order given."""

    def __init__(self, *, technique_order: list[str]):
        self._order = technique_order

    async def select_async(
        self,
        *,
        technique_identifiers,
        objective: str,
        num_top_techniques: int = 1,
        scenario_result_id: str | None = None,
    ):
        return self._order[:num_top_techniques]


@pytest.fixture
def selector():
    return _StubSelector(technique_order=["a", "b", "c"])


@pytest.fixture
def target() -> MagicMock:
    return MagicMock(name="objective_target")


@pytest.fixture
def seed_group() -> SeedAttackGroup:
    return SeedAttackGroup(seeds=[SeedObjective(value="obj")])


class TestInit:
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_empty_techniques(self, target, selector, seed_group):
        with pytest.raises(ValueError, match="techniques"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={},
                selector=selector,
            )

    @pytest.mark.parametrize("bad_max", [0, -1])
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_invalid_max_attempts(self, target, selector, seed_group, bad_max):
        with pytest.raises(ValueError, match="max_attempts_per_objective"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
                selector=selector,
                max_attempts_per_objective=bad_max,
            )


@pytest.mark.usefixtures("patch_central_database")
class TestPerform:
    async def test_stops_on_first_success(self, target, seed_group, monkeypatch):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        selector = _StubSelector(technique_order=["a", "b"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            max_attempts_per_objective=5,
        )
        calls = _patch_child_attack(monkeypatch, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert isinstance(result, SequentialAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert len(calls) == 1

    async def test_retries_until_max_attempts_on_failure(self, target, seed_group, monkeypatch):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.FAILURE]),
            "c": _make_bundle(name="c", outcomes=[AttackOutcome.FAILURE]),
        }
        selector = _StubSelector(technique_order=["a", "b", "c"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            max_attempts_per_objective=3,
        )
        calls = _patch_child_attack(monkeypatch, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.FAILURE
        assert len(calls) == 3

    async def test_passes_attempt_labels_to_inner(self, target, seed_group, monkeypatch):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        selector = _StubSelector(technique_order=["a"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
        )
        calls = _patch_child_attack(monkeypatch, bundles=bundles)

        await dispatcher._perform_async(context=_make_context(labels={"foo": "bar"}))

        labels = calls[0]["attempt_labels"]
        assert labels["foo"] == "bar"
        assert labels[ADAPTIVE_TECHNIQUE_LABEL] == "a"
        assert labels[ADAPTIVE_ATTEMPT_LABEL] == "1"

    async def test_metadata_records_adaptive_trail(self, target, seed_group, monkeypatch):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        selector = _StubSelector(technique_order=["a", "b"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            max_attempts_per_objective=3,
        )
        _patch_child_attack(monkeypatch, bundles=bundles)
        result = await dispatcher._perform_async(context=_make_context())

        trail = result.metadata["adaptive_attempts"]
        assert trail == [
            {"technique": "a", "technique_hash": "a", "outcome": "failure"},
            {"technique": "b", "technique_hash": "b", "outcome": "success"},
        ]

    async def test_envelope_is_distinct_from_child_and_owns_no_conversation(self, target, seed_group, monkeypatch):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        selector = _StubSelector(technique_order=["a"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
        )
        calls = _patch_child_attack(monkeypatch, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert len(calls) == 1
        inner_result = calls[0]["result"]
        # The envelope is a fresh wrapper owning no conversation; the inner
        # attempt's row lives on child_attack_results.
        assert result.attack_result_id != inner_result.attack_result_id
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.conversation_id == ""
        assert result.child_attack_results == [inner_result]
        assert result.metadata["adaptive_attempts"] == [{"technique": "a", "technique_hash": "a", "outcome": "success"}]


@pytest.mark.usefixtures("patch_central_database")
class TestValidate:
    @pytest.mark.parametrize("bad_objective", ["", "   ", "\n\t"])
    def test_validate_rejects_empty_objective(self, target, selector, seed_group, bad_objective):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
        )
        with pytest.raises(ValueError, match="objective"):
            dispatcher._validate_context(context=_make_context(objective=bad_objective))

    def test_validate_accepts_normal_objective(self, target, selector, seed_group):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
        )
        dispatcher._validate_context(context=_make_context(objective="ok"))
