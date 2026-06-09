# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for pyrit.common.notebook_upgrade_helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyrit.common.notebook_upgrade_helper import (
    MIGRATION_RULES,
    MigrationRule,
    UpgradeSuggestion,
    _is_pyrit_traceback,
    _make_attribute_rule,
    _make_import_rule,
    _make_kwarg_rule,
    _make_module_not_found_rule,
    add_migration_rule,
    disable_upgrade_helper,
    enable_upgrade_helper,
    format_suggestion_html,
    format_suggestion_plain,
    is_upgrade_helper_enabled,
    suggest_fix,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _import_error(msg: str) -> ImportError:
    return ImportError(msg)


def _attr_error(msg: str) -> AttributeError:
    return AttributeError(msg)


def _type_error(msg: str) -> TypeError:
    return TypeError(msg)


def _module_not_found_error(msg: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(msg)


PYRIT_TB_LINES = [
    '  File "/home/user/pyrit/score/printer/__init__.py", line 10, in <module>\n'
]
NON_PYRIT_TB_LINES = [
    '  File "/home/user/my_script.py", line 5, in <module>\n'
]


# ── _is_pyrit_traceback ─────────────────────────────────────────────────


class TestIsPyritTraceback:
    def test_detects_pyrit_in_path(self) -> None:
        assert _is_pyrit_traceback(PYRIT_TB_LINES) is True

    def test_no_pyrit_returns_false(self) -> None:
        assert _is_pyrit_traceback(NON_PYRIT_TB_LINES) is False

    def test_empty_returns_false(self) -> None:
        assert _is_pyrit_traceback([]) is False


# ── _make_import_rule ────────────────────────────────────────────────────


class TestMakeImportRule:
    def test_matches_exact_import_error(self) -> None:
        rule = _make_import_rule(
            old_module="pyrit.score.printer",
            old_name="ConsoleScorerPrinter",
            new_module="pyrit.output.scorer.pretty",
            new_name="PrettyScorerMemoryPrinter",
            version="0.15.0",
            description="Moved.",
        )
        exc = _import_error(
            "cannot import name 'ConsoleScorerPrinter' from 'pyrit.score.printer'"
        )
        assert rule.match(exc, []) is True

    def test_no_match_different_name(self) -> None:
        rule = _make_import_rule(
            old_module="pyrit.score.printer",
            old_name="ConsoleScorerPrinter",
            new_module="pyrit.output.scorer.pretty",
            new_name="PrettyScorerMemoryPrinter",
            version="0.15.0",
            description="Moved.",
        )
        exc = _import_error(
            "cannot import name 'SomethingElse' from 'pyrit.score.printer'"
        )
        assert rule.match(exc, []) is False

    def test_defaults_new_name_to_old(self) -> None:
        rule = _make_import_rule(
            old_module="pyrit.foo",
            old_name="Bar",
            new_module="pyrit.baz",
            version="1.0.0",
            description="Moved.",
        )
        assert rule.example_new == "from pyrit.baz import Bar"


# ── _make_module_not_found_rule ──────────────────────────────────────────


class TestMakeModuleNotFoundRule:
    def test_matches_module_not_found(self) -> None:
        rule = _make_module_not_found_rule(
            old_module="pyrit.old_module",
            new_module="pyrit.new_module",
            version="0.15.0",
            description="Module renamed.",
        )
        exc = _module_not_found_error("No module named 'pyrit.old_module'")
        assert rule.match(exc, []) is True

    def test_no_match_different_module(self) -> None:
        rule = _make_module_not_found_rule(
            old_module="pyrit.old_module",
            new_module="pyrit.new_module",
            version="0.15.0",
            description="Module renamed.",
        )
        exc = _module_not_found_error("No module named 'pyrit.something_else'")
        assert rule.match(exc, []) is False


# ── _make_attribute_rule ─────────────────────────────────────────────────


class TestMakeAttributeRule:
    def test_matches_with_pyrit_traceback(self) -> None:
        rule = _make_attribute_rule(
            module="pyrit.models",
            old_name="OldClass",
            new_name="NewClass",
            version="0.15.0",
            description="Renamed.",
        )
        exc = _attr_error("module 'pyrit.models' has no attribute 'OldClass'")
        assert rule.match(exc, PYRIT_TB_LINES) is True

    def test_no_match_without_pyrit_traceback(self) -> None:
        rule = _make_attribute_rule(
            module="pyrit.models",
            old_name="OldClass",
            new_name="NewClass",
            version="0.15.0",
            description="Renamed.",
        )
        exc = _attr_error("module 'pyrit.models' has no attribute 'OldClass'")
        assert rule.match(exc, NON_PYRIT_TB_LINES) is False

    def test_different_module_in_new_ref(self) -> None:
        rule = _make_attribute_rule(
            module="pyrit.old",
            old_name="Foo",
            new_module="pyrit.new",
            new_name="Bar",
            version="1.0.0",
            description="Moved and renamed.",
        )
        assert rule.new_ref == "pyrit.new.Bar"
        assert rule.example_new == "from pyrit.new import Bar"


# ── _make_kwarg_rule ─────────────────────────────────────────────────────


class TestMakeKwargRule:
    def test_matches_with_pyrit_traceback(self) -> None:
        rule = _make_kwarg_rule(
            callable_pattern="MyClass",
            old_kwarg="old_param",
            new_kwarg="new_param",
            version="0.15.0",
            description="Parameter renamed.",
        )
        exc = _type_error(
            "MyClass.__init__() got an unexpected keyword argument 'old_param'"
        )
        assert rule.match(exc, PYRIT_TB_LINES) is True

    def test_no_match_without_pyrit_traceback(self) -> None:
        rule = _make_kwarg_rule(
            callable_pattern="MyClass",
            old_kwarg="old_param",
            new_kwarg="new_param",
            version="0.15.0",
            description="Parameter renamed.",
        )
        exc = _type_error(
            "MyClass.__init__() got an unexpected keyword argument 'old_param'"
        )
        assert rule.match(exc, NON_PYRIT_TB_LINES) is False

    def test_no_match_different_kwarg(self) -> None:
        rule = _make_kwarg_rule(
            callable_pattern="MyClass",
            old_kwarg="old_param",
            new_kwarg="new_param",
            version="0.15.0",
            description="Parameter renamed.",
        )
        exc = _type_error(
            "MyClass.__init__() got an unexpected keyword argument 'something_else'"
        )
        assert rule.match(exc, PYRIT_TB_LINES) is False


# ── suggest_fix ──────────────────────────────────────────────────────────


class TestSuggestFix:
    def test_returns_suggestion_for_known_import_error(self) -> None:
        exc = _import_error(
            "cannot import name 'ConsoleScorerPrinter' from 'pyrit.score.printer'"
        )
        result = suggest_fix(exc)
        assert result is not None
        assert isinstance(result, UpgradeSuggestion)
        assert result.rule.new_ref == "pyrit.output.scorer.pretty.PrettyScorerMemoryPrinter"
        assert result.original_error is exc

    def test_returns_none_for_unknown_error(self) -> None:
        exc = _import_error("cannot import name 'FooBar' from 'some.other.module'")
        result = suggest_fix(exc)
        assert result is None

    def test_returns_none_for_unrelated_exception_type(self) -> None:
        exc = ValueError("something unrelated")
        result = suggest_fix(exc)
        assert result is None

    def test_handles_none_tb_lines(self) -> None:
        exc = _import_error(
            "cannot import name 'AttackResultPrinter' from 'pyrit.executor.attack.printer'"
        )
        result = suggest_fix(exc, tb_lines=None)
        assert result is not None

    def test_broken_rule_matcher_does_not_crash(self) -> None:
        """A rule with a broken matcher should be skipped silently."""
        def broken_matcher(exc: BaseException, tb_lines: list[str]) -> bool:
            raise RuntimeError("Broken!")

        bad_rule = MigrationRule(
            error_types=(ImportError,),
            old_ref="old",
            new_ref="new",
            version_introduced="1.0.0",
            description="Broken rule.",
            example_old="old code",
            example_new="new code",
            match=broken_matcher,
        )

        original_rules = MIGRATION_RULES.copy()
        try:
            MIGRATION_RULES.insert(0, bad_rule)
            exc = _import_error(
                "cannot import name 'ConsoleScorerPrinter' from 'pyrit.score.printer'"
            )
            # Should skip broken rule and still find the real match
            result = suggest_fix(exc)
            assert result is not None
        finally:
            MIGRATION_RULES.clear()
            MIGRATION_RULES.extend(original_rules)


# ── add_migration_rule ───────────────────────────────────────────────────


class TestAddMigrationRule:
    def test_adds_custom_rule(self) -> None:
        original_count = len(MIGRATION_RULES)
        custom_rule = _make_import_rule(
            old_module="mypackage.old",
            old_name="MyClass",
            new_module="mypackage.new",
            version="2.0.0",
            description="Custom move.",
        )
        try:
            add_migration_rule(custom_rule)
            assert len(MIGRATION_RULES) == original_count + 1

            exc = _import_error(
                "cannot import name 'MyClass' from 'mypackage.old'"
            )
            result = suggest_fix(exc)
            assert result is not None
            assert result.rule is custom_rule
        finally:
            MIGRATION_RULES.pop()


# ── format functions ─────────────────────────────────────────────────────


class TestFormatting:
    @pytest.fixture
    def sample_suggestion(self) -> UpgradeSuggestion:
        rule = _make_import_rule(
            old_module="pyrit.score.printer",
            old_name="ConsoleScorerPrinter",
            new_module="pyrit.output.scorer.pretty",
            new_name="PrettyScorerMemoryPrinter",
            version="0.15.0",
            description="Scorer printers moved to pyrit.output.scorer.",
        )
        exc = _import_error("cannot import name 'ConsoleScorerPrinter'")
        return UpgradeSuggestion(rule=rule, original_error=exc)

    def test_plain_format_contains_key_info(self, sample_suggestion: UpgradeSuggestion) -> None:
        text = format_suggestion_plain(sample_suggestion)
        assert "PyRIT Upgrade Suggestion" in text
        assert "v0.15.0" in text
        assert "ConsoleScorerPrinter" in text
        assert "PrettyScorerMemoryPrinter" in text

    def test_html_format_contains_key_info(self, sample_suggestion: UpgradeSuggestion) -> None:
        html = format_suggestion_html(sample_suggestion)
        assert "PyRIT Upgrade Suggestion" in html
        assert "v0.15.0" in html
        assert "ConsoleScorerPrinter" in html
        assert "PrettyScorerMemoryPrinter" in html
        assert "<div" in html


# ── IPython integration ──────────────────────────────────────────────────


class TestIPythonIntegration:
    def setup_method(self) -> None:
        """Reset helper state before each test."""
        import pyrit.common.notebook_upgrade_helper as mod
        mod._is_enabled = False

    def test_enable_returns_false_outside_ipython(self) -> None:
        # get_ipython is not defined outside IPython => NameError => False
        result = enable_upgrade_helper()
        assert result is False
        assert is_upgrade_helper_enabled() is False

    def test_disable_returns_false_when_not_enabled(self) -> None:
        result = disable_upgrade_helper()
        assert result is False

    def test_enable_with_mock_ipython(self) -> None:
        mock_ip = MagicMock()
        with patch("builtins.get_ipython", return_value=mock_ip, create=True):
            result = enable_upgrade_helper()
            assert result is True
            assert is_upgrade_helper_enabled() is True
            mock_ip.set_custom_exc.assert_called_once()

    def test_enable_is_idempotent(self) -> None:
        mock_ip = MagicMock()
        with patch("builtins.get_ipython", return_value=mock_ip, create=True):
            enable_upgrade_helper()
            enable_upgrade_helper()
            # set_custom_exc should only be called once
            assert mock_ip.set_custom_exc.call_count == 1

    def test_disable_with_mock_ipython(self) -> None:
        mock_ip = MagicMock()
        with patch("builtins.get_ipython", return_value=mock_ip, create=True):
            enable_upgrade_helper()
            result = disable_upgrade_helper()
            assert result is True
            assert is_upgrade_helper_enabled() is False
            assert mock_ip.set_custom_exc.call_count == 2  # enable + disable

    def test_exception_handler_shows_traceback_and_suggestion(self) -> None:
        from pyrit.common.notebook_upgrade_helper import _ipython_exception_handler

        mock_shell = MagicMock()
        exc = _import_error(
            "cannot import name 'ConsoleScorerPrinter' from 'pyrit.score.printer'"
        )

        # Call handler with no real traceback
        _ipython_exception_handler(mock_shell, ImportError, exc, None)

        # Verify original traceback was shown
        mock_shell.showtraceback.assert_called_once()

    def test_exception_handler_no_suggestion_for_unmatched(self) -> None:
        from pyrit.common.notebook_upgrade_helper import _ipython_exception_handler

        mock_shell = MagicMock()
        exc = _import_error("cannot import name 'UnknownThing' from 'unknown.module'")

        _ipython_exception_handler(mock_shell, ImportError, exc, None)
        mock_shell.showtraceback.assert_called_once()

    def test_exception_handler_never_crashes(self) -> None:
        """Even if suggest_fix raises, the handler should not propagate."""
        from pyrit.common.notebook_upgrade_helper import _ipython_exception_handler

        mock_shell = MagicMock()
        exc = _import_error("test error")

        with patch(
            "pyrit.common.notebook_upgrade_helper.suggest_fix",
            side_effect=RuntimeError("Unexpected"),
        ):
            # Should not raise
            _ipython_exception_handler(mock_shell, ImportError, exc, None)
            mock_shell.showtraceback.assert_called_once()


# ── Built-in rule coverage ───────────────────────────────────────────────


class TestBuiltInRules:
    """Verify that each built-in rule matches its expected error."""

    # --- Printer import moves (0.15.0) ---
    @pytest.mark.parametrize(
        "error_msg,expected_new_ref",
        [
            (
                "cannot import name 'ConsoleScorerPrinter' from 'pyrit.score.printer'",
                "pyrit.output.scorer.pretty.PrettyScorerMemoryPrinter",
            ),
            (
                "cannot import name 'ScorerPrinter' from 'pyrit.score.printer'",
                "pyrit.output.scorer.base.ScorerPrinterBase",
            ),
            (
                "cannot import name 'ConsoleAttackResultPrinter' from 'pyrit.executor.attack.printer'",
                "pyrit.output.attack_result.pretty.PrettyAttackResultMemoryPrinter",
            ),
            (
                "cannot import name 'AttackResultPrinter' from 'pyrit.executor.attack.printer'",
                "pyrit.output.attack_result.base.AttackResultPrinterBase",
            ),
            (
                "cannot import name 'MarkdownAttackResultPrinter' from 'pyrit.executor.attack.printer'",
                "pyrit.output.attack_result.markdown.MarkdownAttackResultMemoryPrinter",
            ),
        ],
    )
    def test_printer_import_rules(self, error_msg: str, expected_new_ref: str) -> None:
        exc = _import_error(error_msg)
        result = suggest_fix(exc)
        assert result is not None, f"No suggestion for: {error_msg}"
        assert result.rule.new_ref == expected_new_ref

    # --- Orchestrator → Attack renames (0.13.0) ---
    @pytest.mark.parametrize(
        "error_msg,expected_new_ref",
        [
            (
                "cannot import name 'PromptSendingOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.PromptSendingAttack",
            ),
            (
                "cannot import name 'RedTeamingOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.RedTeamingAttack",
            ),
            (
                "cannot import name 'CrescendoOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.CrescendoAttack",
            ),
            (
                "cannot import name 'PAIROrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.PAIRAttack",
            ),
            (
                "cannot import name 'PairOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.PAIRAttack",
            ),
            (
                "cannot import name 'TreeOfAttacksWithPruningOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.TreeOfAttacksWithPruningAttack",
            ),
            (
                "cannot import name 'MultiTurnOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.MultiTurnAttackStrategy",
            ),
            (
                "cannot import name 'FlipAttackOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.FlipAttack",
            ),
            (
                "cannot import name 'SkeletonKeyOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.SkeletonKeyAttack",
            ),
            (
                "cannot import name 'ScoringOrchestrator' from 'pyrit.orchestrator'",
                "pyrit.executor.attack.AttackExecutor",
            ),
        ],
    )
    def test_orchestrator_import_rules(self, error_msg: str, expected_new_ref: str) -> None:
        exc = _import_error(error_msg)
        result = suggest_fix(exc)
        assert result is not None, f"No suggestion for: {error_msg}"
        assert result.rule.new_ref == expected_new_ref

    # --- Module-level: pyrit.orchestrator gone ---
    def test_module_not_found_orchestrator(self) -> None:
        exc = _module_not_found_error("No module named 'pyrit.orchestrator'")
        result = suggest_fix(exc)
        assert result is not None
        assert result.rule.new_ref == "pyrit.executor.attack"

    # --- Kwarg renames (0.13.0) ---
    @pytest.mark.parametrize(
        "error_msg,expected_old_ref",
        [
            (
                "RedTeamingAttack.__init__() got an unexpected keyword argument 'prompt_target'",
                "RedTeamingAttack(prompt_target=...)",
            ),
            (
                "CrescendoAttack.__init__() got an unexpected keyword argument 'red_teaming_chat'",
                "CrescendoAttack(red_teaming_chat=...)",
            ),
            (
                "CrescendoAttack.__init__() got an unexpected keyword argument 'max_rounds'",
                "CrescendoAttack(max_rounds=...)",
            ),
            (
                "PAIRAttack.__init__() got an unexpected keyword argument 'max_conversation_depth'",
                "PAIRAttack(max_conversation_depth=...)",
            ),
            (
                "RedTeamingAttack.__init__() got an unexpected keyword argument 'attack_strategy'",
                "RedTeamingAttack(attack_strategy=...)",
            ),
            (
                "RedTeamingAttack.__init__() got an unexpected keyword argument 'scorer'",
                "RedTeamingAttack(scorer=...)",
            ),
        ],
    )
    def test_kwarg_rules(self, error_msg: str, expected_old_ref: str) -> None:
        exc = _type_error(error_msg)
        result = suggest_fix(exc, tb_lines=PYRIT_TB_LINES)
        assert result is not None, f"No suggestion for: {error_msg}"
        assert result.rule.old_ref == expected_old_ref
