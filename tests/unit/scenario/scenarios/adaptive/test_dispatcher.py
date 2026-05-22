# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup, SeedObjective
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    AdaptiveDispatchAttack,
    AdaptiveDispatchContext,
    AdaptiveDispatchParams,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selectors import (
    EpsilonGreedyTechniqueSelector,
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


def _patch_inner(
    *,
    dispatcher: AdaptiveDispatchAttack,
    bundles: dict[str, TechniqueBundle],
) -> AsyncMock:
    """Replace ``_run_inner_attack_async`` with a stub backed by per-bundle outcomes."""
    name_for_attack = {id(b.attack): name for name, b in bundles.items()}
    counters: dict[str, int] = dict.fromkeys(bundles, 0)

    async def _stub(*, bundle: TechniqueBundle, seed_group, attempt_labels: dict[str, str]) -> AttackResult:
        name = name_for_attack[id(bundle.attack)]
        idx = counters[name]
        counters[name] = idx + 1
        outcome = bundle.attack._outcomes[idx]
        return AttackResult(
            conversation_id=f"conv-{name}-{idx}",
            objective="obj",
            outcome=outcome,
        )

    inner_mock = AsyncMock(side_effect=_stub)
    dispatcher._run_inner_attack_async = inner_mock  # type: ignore[method-assign]
    return inner_mock


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
    async def test_stops_on_first_success(self, target, seed_group):
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
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.SUCCESS
        assert inner.call_count == 1

    async def test_retries_until_max_attempts_on_failure(self, target, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE] * 3),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.FAILURE] * 3),
            "c": _make_bundle(name="c", outcomes=[AttackOutcome.FAILURE] * 3),
        }
        selector = _StubSelector(technique_order=["a", "b", "c"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,


            max_attempts_per_objective=3,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.FAILURE
        assert inner.call_count == 3

    async def test_passes_attempt_labels_to_inner(self, target, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        selector = _StubSelector(technique_order=["a"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,


        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        await dispatcher._perform_async(context=_make_context(labels={"foo": "bar"}))

        labels = inner.call_args.kwargs["attempt_labels"]
        assert labels["foo"] == "bar"
        assert labels[ADAPTIVE_TECHNIQUE_LABEL] == "a"
        assert labels[ADAPTIVE_ATTEMPT_LABEL] == "1"

    async def test_metadata_records_adaptive_trail(self, target, seed_group):
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
        _patch_inner(dispatcher=dispatcher, bundles=bundles)
        result = await dispatcher._perform_async(context=_make_context())

        trail = result.metadata["adaptive_attempts"]
        assert trail == [
            {"technique": "a", "technique_hash": "a", "outcome": "failure"},
            {"technique": "b", "technique_hash": "b", "outcome": "success"},
        ]

    async def test_returns_fresh_result_distinct_from_inner(self, target, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        selector = _StubSelector(technique_order=["a"])
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,


        )
        inner_ids: list[str] = []

        async def _spy(*, bundle, seed_group, attempt_labels):
            inner_result = AttackResult(
                conversation_id="conv-a-0",
                objective="obj",
                outcome=AttackOutcome.SUCCESS,
            )
            inner_ids.append(inner_result.attack_result_id)
            return inner_result

        dispatcher._run_inner_attack_async = AsyncMock(side_effect=_spy)  # type: ignore[method-assign]

        result = await dispatcher._perform_async(context=_make_context())

        assert len(inner_ids) == 1
        assert result.attack_result_id != inner_ids[0]
        assert result.outcome == AttackOutcome.SUCCESS
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
