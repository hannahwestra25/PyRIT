# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for simulated-conversation adversarial prompt-source normalization."""

from pathlib import Path

import pytest

from pyrit.models import SeedPrompt
from pyrit.scenario.core import resolve_simulated_conversation_adversarial_prompt


def test_resolve_simulated_conversation_adversarial_prompt_returns_inline_prompt() -> None:
    prompt = SeedPrompt(value="Use {{ objective }}", parameters=["objective"], data_type="text")

    resolved = resolve_simulated_conversation_adversarial_prompt(
        adversarial_chat_system_prompt=prompt,
    )

    assert resolved is prompt


def test_resolve_simulated_conversation_adversarial_prompt_loads_path(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        "value: Use {{ objective }}\ndata_type: text\nparameters:\n  - objective\nmetadata:\n  source_kind: fixture\n",
        encoding="utf-8",
    )

    resolved = resolve_simulated_conversation_adversarial_prompt(
        adversarial_chat_system_prompt=prompt_path,
    )

    assert resolved.value == "Use {{ objective }}"
    assert resolved.parameters == ["objective"]
    assert resolved.metadata == {"source_kind": "fixture"}
    assert resolved.is_jinja_template is True


def test_resolve_simulated_conversation_adversarial_prompt_accepts_legacy_path_alias(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text("value: Legacy prompt\ndata_type: text\n", encoding="utf-8")

    resolved = resolve_simulated_conversation_adversarial_prompt(
        adversarial_chat_system_prompt_path=prompt_path,
    )

    assert resolved.value == "Legacy prompt"


def test_resolve_simulated_conversation_adversarial_prompt_uses_default_path(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    default_path.write_text("value: Default prompt\ndata_type: text\n", encoding="utf-8")

    resolved = resolve_simulated_conversation_adversarial_prompt(
        default_system_prompt_path=default_path,
    )

    assert resolved.value == "Default prompt"


def test_resolve_simulated_conversation_adversarial_prompt_rejects_both_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only one"):
        resolve_simulated_conversation_adversarial_prompt(
            adversarial_chat_system_prompt_path=tmp_path / "prompt.yaml",
            adversarial_chat_system_prompt=SeedPrompt(value="inline"),
        )


def test_resolve_simulated_conversation_adversarial_prompt_rejects_missing_source() -> None:
    with pytest.raises(ValueError, match="Set one of"):
        resolve_simulated_conversation_adversarial_prompt()
