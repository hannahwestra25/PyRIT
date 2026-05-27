# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses

import pytest

from pyrit.scenario.scenarios.adaptive.selectors import SelectorScope


class TestSelectorScopeDefaults:
    def test_default_constructs_all_runs(self):
        scope = SelectorScope()
        assert scope.current_run_only is False
        assert scope.attack_classes is None
        assert scope.targeted_harm_categories is None
        assert scope.extra_labels is None

    def test_all_runs_classmethod_equivalent_to_default(self):
        assert SelectorScope.all_runs() == SelectorScope()

    def test_current_run_classmethod_sets_flag(self):
        scope = SelectorScope.current_run()
        assert scope.current_run_only is True
        assert scope.attack_classes is None
        assert scope.targeted_harm_categories is None
        assert scope.extra_labels is None


class TestSelectorScopeFrozen:
    def test_assigning_field_raises(self):
        scope = SelectorScope()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.current_run_only = True  # type: ignore[misc]

    def test_assigning_new_field_raises(self):
        scope = SelectorScope()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.extra_labels = {"a": "b"}  # type: ignore[misc]


class TestSelectorScopeCombinations:
    def test_fields_combine(self):
        scope = SelectorScope(
            current_run_only=True,
            attack_classes=["TextAdaptive"],
            targeted_harm_categories=["misinformation"],
            extra_labels={"experiment": "v3"},
        )
        assert scope.current_run_only is True
        assert scope.attack_classes == ["TextAdaptive"]
        assert scope.targeted_harm_categories == ["misinformation"]
        assert scope.extra_labels == {"experiment": "v3"}

    def test_equality_value_based(self):
        a = SelectorScope(attack_classes=("X",), targeted_harm_categories=("y",))
        b = SelectorScope(attack_classes=("X",), targeted_harm_categories=("y",))
        assert a == b

    def test_inequality_when_fields_differ(self):
        a = SelectorScope.all_runs()
        b = SelectorScope.current_run()
        assert a != b
