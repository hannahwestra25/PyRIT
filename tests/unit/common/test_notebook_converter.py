# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for pyrit.common.notebook_converter."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from pyrit.common.notebook_converter import (
    Change,
    ConversionResult,
    convert_notebook,
    convert_source,
)


# ── Import rewriting ─────────────────────────────────────────────────────


class TestImportRewriting:
    def test_orchestrator_single_import(self) -> None:
        source = "from pyrit.orchestrator import CrescendoOrchestrator\n"
        converted, changes = convert_source(source)
        assert "from pyrit.executor.attack import CrescendoAttack" in converted
        assert len(changes) >= 1
        assert changes[0].confidence == "auto"

    def test_orchestrator_multiple_imports(self) -> None:
        source = "from pyrit.orchestrator import CrescendoOrchestrator, PromptSendingOrchestrator\n"
        converted, changes = convert_source(source)
        assert "CrescendoAttack" in converted
        assert "PromptSendingAttack" in converted
        assert "pyrit.executor.attack" in converted

    def test_orchestrator_aliased_import(self) -> None:
        source = "from pyrit.orchestrator import CrescendoOrchestrator as CO\n"
        converted, changes = convert_source(source)
        assert "CrescendoAttack as CO" in converted
        assert "pyrit.executor.attack" in converted

    def test_printer_import_move(self) -> None:
        source = "from pyrit.score.printer import ConsoleScorerPrinter\n"
        converted, changes = convert_source(source)
        assert "from pyrit.output.scorer.pretty import PrettyScorerMemoryPrinter" in converted

    def test_storage_io_move(self) -> None:
        source = "from pyrit.models.storage_io import DiskStorageIO\n"
        converted, changes = convert_source(source)
        assert "from pyrit.memory.storage.storage import DiskStorageIO" in converted

    def test_serializer_move(self) -> None:
        source = "from pyrit.models.data_type_serializer import DataTypeSerializer\n"
        converted, changes = convert_source(source)
        assert "from pyrit.memory.storage.serializers import DataTypeSerializer" in converted

    def test_bare_import_module_rename(self) -> None:
        source = "import pyrit.orchestrator\n"
        converted, changes = convert_source(source)
        assert "import pyrit.executor.attack" in converted
        assert len(changes) == 1

    def test_bare_import_with_alias(self) -> None:
        source = "import pyrit.orchestrator as orch\n"
        converted, changes = convert_source(source)
        assert "import pyrit.executor.attack as orch" in converted

    def test_no_change_for_unrelated_imports(self) -> None:
        source = "from pyrit.prompt_target import AzureOpenAITarget\n"
        converted, changes = convert_source(source)
        assert converted == source
        assert len(changes) == 0

    def test_scorer_identifier_rename(self) -> None:
        source = "from pyrit.models import ScorerIdentifier\n"
        converted, changes = convert_source(source)
        assert "ComponentIdentifier" in converted

    def test_scenario_printer_move(self) -> None:
        source = "from pyrit.scenario.printer import ConsoleScenarioResultPrinter\n"
        converted, changes = convert_source(source)
        assert "from pyrit.output.scenario_result.pretty import PrettyScenarioResultMemoryPrinter" in converted

    def test_split_imports_to_different_modules(self) -> None:
        """When imported names move to different new modules, lines should be split."""
        source = "from pyrit.orchestrator import CrescendoOrchestrator, PromptSendingOrchestrator\n"
        converted, changes = convert_source(source)
        # Both should go to pyrit.executor.attack (same module), single line
        assert converted.count("from pyrit.executor.attack import") == 1


# ── Class name renames ───────────────────────────────────────────────────


class TestClassRenames:
    def test_class_instantiation_renamed(self) -> None:
        source = "orchestrator = CrescendoOrchestrator(target=my_target)\n"
        converted, changes = convert_source(source)
        assert "CrescendoAttack" in converted
        assert "CrescendoOrchestrator" not in converted

    def test_multiple_renames_on_one_line(self) -> None:
        source = "x = PromptSendingOrchestrator if flag else RedTeamingOrchestrator\n"
        converted, changes = convert_source(source)
        assert "PromptSendingAttack" in converted
        assert "RedTeamingAttack" in converted

    def test_word_boundary_prevents_partial_match(self) -> None:
        source = "my_CrescendoOrchestrator_wrapper = True\n"
        converted, changes = convert_source(source)
        # \b treats _ as a word character, so the class name embedded in
        # a longer identifier is correctly NOT renamed
        assert "CrescendoOrchestrator" in converted
        assert len(changes) == 0

    def test_comment_lines_skipped(self) -> None:
        source = "# Using CrescendoOrchestrator for the attack\n"
        converted, changes = convert_source(source)
        assert converted == source  # comments are skipped
        assert len(changes) == 0

    def test_string_content_is_modified(self) -> None:
        """Class names in strings may be intentionally referencing the class."""
        source = 'name = "CrescendoOrchestrator"\n'
        converted, changes = convert_source(source)
        # This IS modified — users may want to update string references too
        assert "CrescendoAttack" in converted


# ── Kwarg renames ────────────────────────────────────────────────────────


class TestKwargRenames:
    def test_prompt_target_renamed_with_context(self) -> None:
        source = textwrap.dedent("""\
            attack = RedTeamingAttack(
                prompt_target=my_target,
            )
        """)
        converted, changes = convert_source(source)
        assert "objective_target=my_target" in converted

    def test_red_teaming_chat_renamed(self) -> None:
        source = "attack = CrescendoAttack(red_teaming_chat=chat_model)\n"
        converted, changes = convert_source(source)
        assert "adversarial_chat=chat_model" in converted

    def test_max_rounds_renamed(self) -> None:
        source = "attack = CrescendoAttack(max_rounds=10)\n"
        converted, changes = convert_source(source)
        assert "max_turns=10" in converted

    def test_max_conversation_depth_renamed(self) -> None:
        source = "attack = PAIRAttack(max_conversation_depth=5)\n"
        converted, changes = convert_source(source)
        assert "max_turns=5" in converted

    def test_scorer_kwarg_is_suggestion(self) -> None:
        source = "attack = RedTeamingAttack(scorer=my_scorer)\n"
        converted, changes = convert_source(source)
        assert "objective_scorer=my_scorer" in converted
        scorer_change = [c for c in changes if "scorer" in c.description]
        assert scorer_change
        assert scorer_change[0].confidence == "suggestion"

    def test_kwarg_not_renamed_without_context(self) -> None:
        source = "result = my_function(prompt_target=foo)\n"
        converted, changes = convert_source(source)
        # No Attack/Orchestrator context → should NOT be rewritten
        assert "prompt_target=foo" in converted

    def test_attack_strategy_renamed(self) -> None:
        source = "attack = RedTeamingAttack(attack_strategy='be sneaky')\n"
        converted, changes = convert_source(source)
        assert "objective='be sneaky'" in converted


# ── Method renames ───────────────────────────────────────────────────────


class TestMethodRenames:
    def test_print_conversation_async_renamed(self) -> None:
        source = "await attack.print_conversation_async()\n"
        converted, changes = convert_source(source)
        assert ".write_async()" in converted
        assert changes[0].confidence == "auto"

    def test_normalize_strategies_renamed(self) -> None:
        source = "strategies = scenario.normalize_strategies(data)\n"
        converted, changes = convert_source(source)
        assert ".expand(data)" in converted

    def test_to_dict_is_suggestion(self) -> None:
        source = "result = AttackResult.from_dict(data)\n"
        converted, changes = convert_source(source)
        assert ".model_validate(data)" in converted
        assert changes[0].confidence == "suggestion"

    def test_to_dict_not_renamed_without_context(self) -> None:
        source = "data = my_object.to_dict()\n"
        converted, changes = convert_source(source)
        # No AttackResult/MessagePiece/Score context → should NOT be rewritten
        assert ".to_dict()" in converted


# ── Full notebook conversion ─────────────────────────────────────────────


class TestNotebookConversion:
    def test_ipynb_code_cells_converted(self, tmp_path: Path) -> None:
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Using CrescendoOrchestrator\n"],
                },
                {
                    "cell_type": "code",
                    "source": [
                        "from pyrit.orchestrator import CrescendoOrchestrator\n",
                        "\n",
                        "attack = CrescendoOrchestrator(prompt_target=target)\n",
                    ],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        result = convert_notebook(nb_path)
        assert result.has_changes()

        # Check that markdown cells were NOT modified
        converted_nb = json.loads(result.converted_source)
        md_cell = converted_nb["cells"][0]
        assert "CrescendoOrchestrator" in md_cell["source"][0]

        # Check that code cells WERE modified
        code_cell = converted_nb["cells"][1]
        code = "".join(code_cell["source"])
        assert "CrescendoAttack" in code
        assert "pyrit.executor.attack" in code

    def test_ipynb_markdown_cells_untouched(self, tmp_path: Path) -> None:
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["Use PromptSendingOrchestrator to send prompts.\n"],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        result = convert_notebook(nb_path)
        assert not result.has_changes()

    def test_py_file_conversion(self, tmp_path: Path) -> None:
        source = textwrap.dedent("""\
            from pyrit.orchestrator import RedTeamingOrchestrator

            orch = RedTeamingOrchestrator(
                prompt_target=target,
                red_teaming_chat=chat,
                max_rounds=10,
            )
            await orch.print_conversation_async()
        """)
        py_path = tmp_path / "test.py"
        py_path.write_text(source, encoding="utf-8")

        result = convert_notebook(py_path)
        assert result.has_changes()
        assert "RedTeamingAttack" in result.converted_source
        assert "objective_target=target" in result.converted_source
        assert "adversarial_chat=chat" in result.converted_source
        assert "max_turns=10" in result.converted_source
        assert ".write_async()" in result.converted_source

    def test_write_output(self, tmp_path: Path) -> None:
        source = "from pyrit.orchestrator import CrescendoOrchestrator\n"
        py_path = tmp_path / "test.py"
        py_path.write_text(source, encoding="utf-8")

        result = convert_notebook(py_path, write=True)
        written = py_path.read_text(encoding="utf-8")
        assert "CrescendoAttack" in written

    def test_write_to_different_path(self, tmp_path: Path) -> None:
        source = "from pyrit.orchestrator import CrescendoOrchestrator\n"
        py_path = tmp_path / "test.py"
        py_path.write_text(source, encoding="utf-8")

        out_path = tmp_path / "converted.py"
        result = convert_notebook(py_path, write=True, output_path=out_path)
        assert out_path.exists()
        assert "CrescendoAttack" in out_path.read_text(encoding="utf-8")
        # Original should be unchanged
        assert "CrescendoOrchestrator" in py_path.read_text(encoding="utf-8")

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            convert_notebook("nonexistent.py")

    def test_no_changes_returns_clean_result(self, tmp_path: Path) -> None:
        source = "from pyrit.prompt_target import AzureOpenAITarget\n"
        py_path = tmp_path / "test.py"
        py_path.write_text(source, encoding="utf-8")

        result = convert_notebook(py_path)
        assert not result.has_changes()
        assert "up to date" in result.summary()


# ── ConversionResult ─────────────────────────────────────────────────────


class TestConversionResult:
    def test_diff_output(self) -> None:
        result = ConversionResult(
            original_source="from pyrit.orchestrator import CrescendoOrchestrator\n",
            converted_source="from pyrit.executor.attack import CrescendoAttack\n",
            changes=[
                Change(
                    line_number=1,
                    description="test",
                    old_text="old",
                    new_text="new",
                    confidence="auto",
                    category="import",
                )
            ],
            file_path="test.py",
        )
        diff = result.diff()
        assert "-from pyrit.orchestrator" in diff
        assert "+from pyrit.executor.attack" in diff

    def test_summary_counts(self) -> None:
        result = ConversionResult(
            original_source="",
            converted_source="",
            changes=[
                Change(1, "a", "", "", "auto", "import"),
                Change(2, "b", "", "", "suggestion", "kwarg"),
                Change(3, "c", "", "", "auto", "class_rename"),
            ],
        )
        summary = result.summary()
        assert "3 change(s)" in summary
        assert "2 auto-applied" in summary
        assert "1 suggestion" in summary

    def test_write_raises_without_path(self) -> None:
        result = ConversionResult(
            original_source="",
            converted_source="",
        )
        with pytest.raises(ValueError):
            result.write()


# ── End-to-end realistic scenario ────────────────────────────────────────


class TestEndToEnd:
    def test_full_old_notebook_conversion(self, tmp_path: Path) -> None:
        """Simulate a real old-style PyRIT notebook."""
        old_code = textwrap.dedent("""\
            from pyrit.orchestrator import CrescendoOrchestrator
            from pyrit.orchestrator import PromptSendingOrchestrator
            from pyrit.score.printer import ConsoleScorerPrinter
            from pyrit.models.storage_io import DiskStorageIO

            # Set up the attack
            attack = CrescendoOrchestrator(
                prompt_target=my_target,
                red_teaming_chat=my_chat,
                max_rounds=10,
                scorer=my_scorer,
            )

            result = await attack.run_attack_async(objective="test")
            await attack.print_conversation_async()

            # Print scores
            printer = ConsoleScorerPrinter()
        """)

        py_path = tmp_path / "old_notebook.py"
        py_path.write_text(old_code, encoding="utf-8")

        result = convert_notebook(py_path)
        converted = result.converted_source

        # Verify imports updated
        assert "from pyrit.executor.attack import CrescendoAttack" in converted
        assert "from pyrit.executor.attack import PromptSendingAttack" in converted
        assert "from pyrit.output.scorer.pretty import PrettyScorerMemoryPrinter" in converted
        assert "from pyrit.memory.storage.storage import DiskStorageIO" in converted

        # Verify class names updated
        assert "CrescendoAttack(" in converted
        assert "PrettyScorerMemoryPrinter()" in converted

        # Verify kwargs updated
        assert "objective_target=my_target" in converted
        assert "adversarial_chat=my_chat" in converted
        assert "max_turns=10" in converted

        # Verify method updated
        assert ".write_async()" in converted

        # Verify old names are gone
        assert "CrescendoOrchestrator" not in converted
        assert "pyrit.orchestrator" not in converted
        assert "ConsoleScorerPrinter" not in converted
        assert "prompt_target=" not in converted
        assert "red_teaming_chat=" not in converted
        assert "max_rounds=" not in converted
