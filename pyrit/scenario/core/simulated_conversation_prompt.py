# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Prompt-source normalization for simulated-conversation attack techniques."""

from pathlib import Path

from pyrit.models import SeedPrompt


def resolve_simulated_conversation_adversarial_prompt(
    *,
    adversarial_chat_system_prompt_path: str | Path | None = None,
    adversarial_chat_system_prompt: SeedPrompt | None = None,
    default_system_prompt_path: str | Path | None = None,
) -> SeedPrompt:
    """
    Resolve one simulated-conversation adversarial prompt source to a ``SeedPrompt``.

    An explicit path and inline prompt are mutually exclusive. When neither is
    explicit, ``default_system_prompt_path`` supplies the factory's conventional
    name-based YAML fallback.

    Args:
        adversarial_chat_system_prompt_path: Explicit YAML prompt path.
        adversarial_chat_system_prompt: Explicit inline prompt.
        default_system_prompt_path: YAML fallback used when no explicit source is provided.

    Returns:
        The canonical inline prompt.

    Raises:
        ValueError: If both sources are provided or no source/default is available.
    """
    if adversarial_chat_system_prompt_path is not None and adversarial_chat_system_prompt is not None:
        raise ValueError("Set only one of adversarial_chat_system_prompt_path or adversarial_chat_system_prompt.")

    if adversarial_chat_system_prompt is not None:
        return adversarial_chat_system_prompt

    prompt_path = (
        adversarial_chat_system_prompt_path
        if adversarial_chat_system_prompt_path is not None
        else default_system_prompt_path
    )
    if prompt_path is None:
        raise ValueError(
            "Set one of adversarial_chat_system_prompt_path or adversarial_chat_system_prompt, "
            "or provide default_system_prompt_path."
        )

    return SeedPrompt.from_yaml_file(prompt_path)
