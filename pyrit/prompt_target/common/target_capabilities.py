# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, fields
from typing import Optional

from pyrit.models import PromptDataType


@dataclass(frozen=True)
class TargetCapabilities:
    """
    Describes the capabilities of a PromptTarget so that attacks
    and other components can adapt their behavior accordingly.

    Each target class defines default capabilities via the _DEFAULT_CAPABILITIES
    class attribute. Users can override individual capabilities per instance
    through constructor parameters, which is useful for targets whose
    capabilities depend on deployment configuration (e.g., Playwright, HTTP).
    """

    # Whether the target natively supports multi-turn conversations
    # (i.e., it accepts and uses conversation history or maintains state
    # across turns via external mechanisms like WebSocket connections).
    supports_multi_turn: bool = False

    # Whether the target natively supports multiple message pieces in a single request.
    supports_multi_message_pieces: bool = False

    # Whether the target natively supports constraining output to a provided JSON schema.
    supports_json_schema: bool = False

    # Whether the target natively supports JSON output (e.g., via a "json" response format), which ensures the output
    # is valid JSON.
    supports_json_output: bool = False

    # Whether the target allows the attack history to be modified
    supports_editable_history: bool = False

    # The input modalities supported by the target (e.g., "text", "image").
    input_modalities: frozenset[frozenset[PromptDataType]] = frozenset({frozenset(["text"])})

    # The output modalities supported by the target (e.g., "text", "image").
    output_modalities: frozenset[frozenset[PromptDataType]] = frozenset({frozenset(["text"])})

    def assert_satifies(self, required_capabilities: "TargetCapabilities") -> None:
        """
        Assert that the current capabilities satisfy the required capabilities.

        Args:
            required_capabilities (TargetCapabilities): The required capabilities to check against.

        Raises:
            ValueError: If any of the required capabilities are not satisfied.
        """
        unmet = []
        for f in fields(required_capabilities):
            required_value = getattr(required_capabilities, f.name)
            self_value = getattr(self, f.name)
            if (
                isinstance(required_value, frozenset)
                and required_value
                and isinstance(next(iter(required_value)), frozenset)
            ):
                missing = required_value - self_value
                if missing:
                    unmet.append(f"{f.name}: missing {missing}")
            elif required_value and not self_value:
                unmet.append(f.name)
        if unmet:
            raise ValueError(f"Target does not satisfy the following capabilities: {', '.join(unmet)}")

    @staticmethod
    def get_known_capabilities(underlying_model: str) -> "Optional[TargetCapabilities]":
        """
        Return the known capabilities for a specific underlying model, or None if unrecognized.

        Args:
            underlying_model (str): The underlying model name (e.g., "gpt-4o").

        Returns:
            TargetCapabilities | None: The known capabilities for the model, or None if the model
            is not recognized.
        """
        if underlying_model == "gpt-4o":
            return TargetCapabilities(
                supports_multi_turn=True,
                supports_multi_message_pieces=True,
                supports_json_output=True,
                input_modalities=frozenset(
                    {
                        frozenset({"text"}),
                        frozenset({"image_path"}),
                        frozenset({"text", "image_path"}),
                    }
                ),
                output_modalities=frozenset(
                    {
                        frozenset({"text"}),
                    }
                ),
            )
        if underlying_model in ["gpt-5.4", "gpt-5.1", "gpt-5", "gpt-5.4-mini"]:
            return TargetCapabilities(
                supports_multi_turn=True,
                supports_multi_message_pieces=True,
                supports_json_schema=True,
                supports_json_output=True,
                input_modalities=frozenset(
                    {frozenset({"text", "image_path"}), frozenset({"image_path"}), frozenset({"text"})}
                ),
                output_modalities=frozenset({frozenset({"text"})}),
            )
        if underlying_model == "gpt-realtime-1.5":
            return TargetCapabilities(
                supports_multi_turn=True,
                supports_multi_message_pieces=True,
                supports_editable_history=True,
                input_modalities=frozenset(
                    {
                        frozenset({"text"}),
                        frozenset({"audio_path"}),
                        frozenset({"image_path"}),
                        frozenset({"text", "audio_path"}),
                        frozenset({"text", "image_path"}),
                        frozenset({"audio_path", "image_path"}),
                        frozenset({"text", "audio_path", "image_path"}),
                    }
                ),
                output_modalities=frozenset(
                    {
                        frozenset({"text"}),
                        frozenset({"audio_path"}),
                        frozenset({"text", "audio_path"}),
                    }
                ),
            )
        if underlying_model == "tts":
            return TargetCapabilities(
                output_modalities=frozenset({frozenset({"audio_path"})}),
            )
        if underlying_model == "sora-2":
            return TargetCapabilities(
                supports_multi_turn=True,
                supports_multi_message_pieces=True,
                input_modalities=frozenset(
                    {frozenset({"text", "image_path"}), frozenset({"image_path"}), frozenset({"text"})}
                ),
                output_modalities=frozenset({frozenset({"audio_path", "video_path"}), frozenset({"video_path"})}),
            )

        return None
