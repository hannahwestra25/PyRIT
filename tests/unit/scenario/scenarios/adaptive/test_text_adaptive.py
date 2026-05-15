# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ``TextAdaptive`` scenario."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.models import SeedAttackGroup, SeedObjective
from pyrit.prompt_target import PromptTarget
from pyrit.registry.object_registries.attack_technique_registry import AttackTechniqueRegistry
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselinePolicy
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    BANDIT_CONTEXT_LABEL,
    AdaptiveDispatchAttack,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
    harm_category_context,
)
from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive
from pyrit.score import TrueFalseScorer


_MOCK_MANY_SHOT_EXAMPLES = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(100)]


def _mock_id(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="test")


@pytest.fixture
def mock_objective_target() -> MagicMock:
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = _mock_id("MockObjectiveTarget")
    return mock


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    mock = MagicMock(spec=TrueFalseScorer)
    mock.get_identifier.return_value = _mock_id("MockObjectiveScorer")
    return mock


@pytest.fixture(autouse=True)
def reset_technique_registry():
    """Reset registries and the cached strategy class between tests."""
    from pyrit.registry import TargetRegistry

    AttackTechniqueRegistry.reset_instance()
    TargetRegistry.reset_instance()
    TextAdaptive._cached_strategy_class = None
    yield
    AttackTechniqueRegistry.reset_instance()
    TargetRegistry.reset_instance()
    TextAdaptive._cached_strategy_class = None


@pytest.fixture(autouse=True)
def patch_many_shot_load():
    with patch(
        "pyrit.executor.attack.single_turn.many_shot_jailbreak.load_many_shot_jailbreaking_dataset",
        return_value=_MOCK_MANY_SHOT_EXAMPLES,
    ):
        yield


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "gpt-4",
        },
    ):
        yield


def _make_seed_group(*, value: str, harm_categories: list[str] | None = None) -> SeedAttackGroup:
    return SeedAttackGroup(seeds=[SeedObjective(value=value, harm_categories=harm_categories)])


FIXTURES = ["patch_central_database", "mock_runtime_env"]


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveBasics:
    def test_version(self):
        assert TextAdaptive.VERSION == 1

    def test_baseline_forbidden(self):
        assert TextAdaptive.BASELINE_POLICY is BaselinePolicy.Forbidden

    def test_default_dataset_config(self):
        config = TextAdaptive.default_dataset_config()
        assert isinstance(config, DatasetConfiguration)
        assert config.max_dataset_size == 4

    def test_required_datasets_non_empty(self):
        assert len(TextAdaptive.required_datasets()) > 0

    def test_get_strategy_class_is_cached(self):
        cls_a = TextAdaptive.get_strategy_class()
        cls_b = TextAdaptive.get_strategy_class()
        assert cls_a is cls_b

    def test_get_default_strategy(self):
        strat = TextAdaptive.get_default_strategy()
        # The default aggregate must resolve to something runnable.
        assert strat is not None

    @patch("pyrit.scenario.core.scenario.Scenario._get_default_objective_scorer")
    def test_init_stores_bandit_params(self, mock_get_scorer, mock_objective_scorer):
        mock_get_scorer.return_value = mock_objective_scorer
        scenario = TextAdaptive(
            epsilon=0.4,
            pool_threshold=5,
            max_attempts_per_objective=7,
            seed=42,
        )
        assert scenario._epsilon == 0.4
        assert scenario._pool_threshold == 5
        assert scenario._max_attempts_per_objective == 7
        assert scenario._seed == 42


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveAtomicAttacks:
    """Tests for ``_get_atomic_attacks_async`` overriding."""

    async def _build_scenario_and_attacks(
        self,
        *,
        mock_objective_target,
        mock_objective_scorer,
        seed_groups: dict[str, list[SeedAttackGroup]],
        **scenario_kwargs,
    ):
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=seed_groups):
            scenario = TextAdaptive(
                objective_scorer=mock_objective_scorer,
                **scenario_kwargs,
            )
            await scenario.initialize_async(
                objective_target=mock_objective_target,
                include_baseline=False,
            )
            return scenario, await scenario._get_atomic_attacks_async()

    async def test_one_atomic_per_objective(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [
                _make_seed_group(value="obj-v1", harm_categories=["violence"]),
                _make_seed_group(value="obj-v2", harm_categories=["violence"]),
            ],
            "hate": [
                _make_seed_group(value="obj-h1", harm_categories=["hate"]),
            ],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        assert len(attacks) == 3
        for atomic in attacks:
            # Each atomic carries exactly one seed group.
            assert len(atomic.objectives) == 1

    async def test_all_atomics_share_one_dispatcher(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [
                _make_seed_group(value="obj-v1", harm_categories=["violence"]),
                _make_seed_group(value="obj-v2", harm_categories=["violence"]),
            ],
        }
        scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        dispatchers = {atomic._attack_technique.attack for atomic in attacks}
        assert len(dispatchers) == 1
        assert isinstance(next(iter(dispatchers)), AdaptiveDispatchAttack)
        assert isinstance(scenario._selector, AdaptiveTechniqueSelector)

    async def test_global_context_label_when_using_global_extractor(
        self, mock_objective_target, mock_objective_scorer
    ):
        groups = {
            "violence": [_make_seed_group(value="obj-1", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-2", harm_categories=["hate"])],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        for atomic in attacks:
            assert atomic._memory_labels[BANDIT_CONTEXT_LABEL] == GLOBAL_CONTEXT

    async def test_harm_category_extractor_partitions_labels(
        self, mock_objective_target, mock_objective_scorer
    ):
        groups = {
            "violence": [_make_seed_group(value="obj-v", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-h", harm_categories=["hate"])],
            "uncat": [_make_seed_group(value="obj-u", harm_categories=None)],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
            context_extractor=harm_category_context,
        )
        contexts = {atomic._memory_labels[BANDIT_CONTEXT_LABEL] for atomic in attacks}
        # Each objective gets its own context bucket from harm_category_context.
        assert contexts == {"violence", "hate", "_uncategorized"}

    async def test_atomic_names_are_unique(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [
                _make_seed_group(value=f"obj-{i}", harm_categories=["violence"]) for i in range(5)
            ],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        names = [atomic.atomic_attack_name for atomic in attacks]
        assert len(set(names)) == len(names)

    async def test_display_group_is_dataset_name(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [_make_seed_group(value="obj-v", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-h", harm_categories=["hate"])],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        display_groups = {atomic.display_group for atomic in attacks}
        assert display_groups == {"violence", "hate"}

    async def test_no_usable_techniques_raises(self, mock_objective_target, mock_objective_scorer):
        groups = {"violence": [_make_seed_group(value="obj")]}
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(
                objective_target=mock_objective_target,
                include_baseline=False,
            )
            # Force the factory map to be empty.
            with patch.object(scenario, "_get_attack_technique_factories", return_value={}):
                with pytest.raises(ValueError, match="no usable techniques"):
                    await scenario._get_atomic_attacks_async()


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveBaselinePolicy:
    async def test_initialize_async_rejects_explicit_baseline(
        self, mock_objective_target, mock_objective_scorer
    ):
        groups = {"violence": [_make_seed_group(value="obj", harm_categories=["violence"])]}
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            with pytest.raises(ValueError):
                await scenario.initialize_async(
                    objective_target=mock_objective_target,
                    include_baseline=True,
                )
