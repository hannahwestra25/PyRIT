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

    Each invocation records the merged ``memory_labels``, the forwarded
    ``attribution``, and the resulting ``AttackResult`` so tests can inspect
    per-attempt routing, per-attempt label stamping, and attribution
    propagation without monkey-patching ``AttackExecutor``.
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
                "attribution": attribution,
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

    async def test_attribution_forwarded_to_inner_sequence(self, target, seed_group, monkeypatch):
        """
        The outer dispatcher owns persistence; child attacks need the scenario
        ``_attribution`` forwarded onto the inner SequentialAttack context so
        per-child rows persist with the scenario linkage. Without this, the
        per-dataset adaptive results disappear from per-scenario hydration.
        """
        from pyrit.executor.attack.core.attack_result_attribution import AttackResultAttribution

        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=_StubSelector(technique_order=["a"]),
        )
        calls = _patch_child_attack(monkeypatch, bundles=bundles)

        ctx = _make_context()
        attribution = AttackResultAttribution(
            parent_id="scenario-1",
            parent_collection="adaptive_airt_hate",
        )
        ctx._attribution = attribution

        await dispatcher._perform_async(context=ctx)

        assert calls[0]["attribution"] is attribution

    async def test_inner_lifecycle_bypassed_no_double_persist(self, target, seed_group, monkeypatch):
        """
        The outer dispatcher must drive the inner SequentialAttack through
        ``_perform_async`` directly, never through ``execute_async``. The
        latter triggers the inner ``_on_post_execute`` which persists the
        same ``attack_result_id`` a second time and rolls the existing row's
        attribution off, hiding per-dataset adaptive results from
        per-scenario hydration.
        """
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=_StubSelector(technique_order=["a"]),
        )
        _patch_child_attack(monkeypatch, bundles=bundles)

        async def _boom(self, *args, **kwargs):
            raise AssertionError(
                "Dispatcher must not call SequentialAttack.execute_async — "
                "doing so triggers double-persistence of the envelope."
            )

        monkeypatch.setattr(SequentialAttack, "execute_async", _boom, raising=True)

        result = await dispatcher._perform_async(context=_make_context())
        assert result.outcome == AttackOutcome.SUCCESS


@pytest.mark.usefixtures("patch_central_database")
class TestEndToEndPersistence:
    """
    Drive the full ``execute_with_context_async`` lifecycle against a real
    in-memory SQLite to verify the dispatcher persists the envelope exactly
    once and stamps it with the outer ``AttackResultAttribution``. This is
    the integration-shape test that catches the bug where the inner
    SequentialAttack's lifecycle would race the outer dispatcher's
    persistence and strip attribution off the row.
    """

    async def _seed_scenario_row(self, sqlite_instance):
        """Insert a minimal ScenarioResultEntry so the FK on attribution_parent_id can land."""
        from pyrit.models import ScenarioIdentifier, ScenarioResult

        scenario = ScenarioResult(
            scenario_identifier=ScenarioIdentifier(name="test_scenario"),
            objective_target_identifier=None,
            objective_scorer_identifier=None,
            attack_results={"adaptive_airt_hate": []},
            scenario_run_state="CREATED",
            display_group_map={"adaptive_airt_hate": "airt_hate"},
        )
        sqlite_instance.add_scenario_results_to_memory(scenario_results=[scenario])
        return str(scenario.id)

    async def test_envelope_persisted_once_with_attribution(self, target, monkeypatch, sqlite_instance):
        """
        End-to-end: drive ``dispatcher.execute_with_context_async`` and assert
        exactly one envelope row lands in the DB with attribution stamped to
        the outer scenario. Catches the regression where the inner
        SequentialAttack's lifecycle either double-persists (IntegrityError
        rollback strips attribution) or skips attribution forwarding.
        """
        from pyrit.executor.attack.core.attack_result_attribution import AttackResultAttribution
        from pyrit.memory.memory_models import AttackResultEntry

        scenario_id = await self._seed_scenario_row(sqlite_instance)
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=_StubSelector(technique_order=["a"]),
        )
        _patch_child_attack(monkeypatch, bundles=bundles)

        ctx = _make_context()
        ctx._attribution = AttackResultAttribution(
            parent_id=scenario_id,
            parent_collection="adaptive_airt_hate",
        )

        await dispatcher.execute_with_context_async(context=ctx)

        with sqlite_instance.get_session() as session:
            rows = session.query(AttackResultEntry).all()

        envelopes = [r for r in rows if str(r.attribution_parent_id) == scenario_id]
        assert len(envelopes) == 1, (
            f"Expected exactly 1 attributed envelope row; found {len(envelopes)}. "
            f"All rows: {[(r.id, r.attribution_parent_id, r.attribution_data) for r in rows]}"
        )
        envelope = envelopes[0]
        assert envelope.attribution_data["parent_collection"] == "adaptive_airt_hate"

    async def test_full_lifecycle_with_real_prompt_sending_attack(self, sqlite_instance):
        """
        End-to-end with a real ``PromptSendingAttack`` child (no stubs) running
        against a ``MockPromptTarget``. Verifies the full dispatcher → sequential
        → child → memory chain works without IntegrityErrors and that both
        envelope and child rows land in the DB with attribution. This is the
        shape that exercises the path the live backend actually takes.
        """
        from pyrit.executor.attack.core.attack_result_attribution import AttackResultAttribution
        from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
        from pyrit.memory.memory_models import AttackResultEntry
        from pyrit.scenario.scenarios.adaptive.dispatcher import TechniqueBundle
        from tests.unit.mocks import MockPromptTarget

        scenario_id = await self._seed_scenario_row(sqlite_instance)

        live_target = MockPromptTarget()
        child_attack = PromptSendingAttack(objective_target=live_target)
        bundles = {"role_play": TechniqueBundle(attack=child_attack, name="role_play")}

        dispatcher = AdaptiveDispatchAttack(
            objective_target=live_target,
            techniques=bundles,
            selector=_StubSelector(technique_order=["role_play"]),
            max_attempts_per_objective=1,
        )

        ctx = _make_context()
        ctx._attribution = AttackResultAttribution(
            parent_id=scenario_id,
            parent_collection="adaptive_airt_hate",
        )

        await dispatcher.execute_with_context_async(context=ctx)

        with sqlite_instance.get_session() as session:
            rows = session.query(AttackResultEntry).all()

        attributed = [r for r in rows if str(r.attribution_parent_id) == scenario_id]
        assert len(attributed) >= 1, (
            f"Expected at least one AttackResultEntry attributed to scenario {scenario_id}; "
            f"found {len(attributed)} (total rows: {len(rows)})."
        )
        for row in attributed:
            assert (row.attribution_data or {}).get("parent_collection") == "adaptive_airt_hate"


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
