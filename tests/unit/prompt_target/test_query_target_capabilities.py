# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pyrit.models import Message, MessagePiece, PromptDataType
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.query_target_capabilities import (
    _CAPABILITY_PROBES,
    _create_test_message,
    _permissive_configuration,
    query_target_capabilities_async,
    verify_target_modalities_async,
)
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityName,
    TargetCapabilities,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

from tests.unit.mocks import MockPromptTarget


def _ok_response(*, conversation_id: str = "probe", text: str = "ok") -> list[Message]:
    return [
        Message(
            [
                MessagePiece(
                    role="assistant",
                    original_value=text,
                    original_value_data_type="text",
                    conversation_id=conversation_id,
                    response_error="none",
                )
            ]
        )
    ]


def _error_response(*, conversation_id: str = "probe") -> list[Message]:
    return [
        Message(
            [
                MessagePiece(
                    role="assistant",
                    original_value="blocked",
                    original_value_data_type="text",
                    conversation_id=conversation_id,
                    response_error="blocked",
                )
            ]
        )
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestPermissiveConfiguration:
    def test_replaces_and_restores_configuration(self) -> None:
        target = MockPromptTarget()
        original = target.configuration

        with _permissive_configuration(target=target):
            permissive = target.configuration
            assert permissive is not original
            for capability in CapabilityName:
                assert permissive.includes(capability=capability)

        assert target.configuration is original

    def test_restores_on_exception(self) -> None:
        target = MockPromptTarget()
        original = target.configuration

        with pytest.raises(RuntimeError):
            with _permissive_configuration(target=target):
                raise RuntimeError("boom")

        assert target.configuration is original


@pytest.mark.usefixtures("patch_central_database")
class TestQueryTargetCapabilitiesAsync:
    async def test_returns_only_supported_when_all_probes_succeed(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await query_target_capabilities_async(target=target)

        # Every capability with a probe should be in the result.
        for capability in _CAPABILITY_PROBES:
            assert capability in result

    async def test_excludes_capabilities_when_probe_fails(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(side_effect=Exception("nope"))

        result = await query_target_capabilities_async(target=target)

        for capability in _CAPABILITY_PROBES:
            assert capability not in result

    async def test_excludes_capabilities_when_response_has_error(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_error_response())

        result = await query_target_capabilities_async(target=target)

        for capability in _CAPABILITY_PROBES:
            assert capability not in result

    async def test_filters_by_requested_capabilities(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        requested = {CapabilityName.SYSTEM_PROMPT, CapabilityName.MULTI_TURN}
        result = await query_target_capabilities_async(target=target, capabilities=requested)

        assert result == requested

    async def test_capability_without_probe_falls_back_to_declared_support(self) -> None:
        target = MockPromptTarget()
        # Override the configuration so editable_history is declared as supported.
        target._configuration = TargetConfiguration(
            capabilities=TargetCapabilities(supports_editable_history=True),
        )
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.EDITABLE_HISTORY},
        )

        assert result == {CapabilityName.EDITABLE_HISTORY}

    async def test_capability_without_probe_excluded_when_not_declared(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.EDITABLE_HISTORY},
        )

        assert result == set()

    async def test_restores_configuration_after_probing(self) -> None:
        target = MockPromptTarget()
        original = target.configuration
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        await query_target_capabilities_async(target=target)

        assert target.configuration is original

    async def test_multi_turn_probe_makes_two_calls_with_same_conversation_id(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_TURN},
        )

        # Multi-turn probe sends two messages on the same conversation_id.
        calls = target.send_prompt_async.await_args_list
        assert len(calls) == 2
        first_conv_id = calls[0].kwargs["message"].message_pieces[0].conversation_id
        second_conv_id = calls[1].kwargs["message"].message_pieces[0].conversation_id
        assert first_conv_id == second_conv_id

    async def test_multi_turn_probe_short_circuits_on_first_failure(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(side_effect=Exception("first call fails"))

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_TURN},
        )

        assert result == set()
        assert target.send_prompt_async.await_count == 1

    async def test_json_schema_probe_sends_schema_in_metadata(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.JSON_SCHEMA},
        )

        message: Message = target.send_prompt_async.await_args.kwargs["message"]
        metadata = message.message_pieces[0].prompt_metadata
        assert metadata is not None
        assert metadata["response_format"] == "json"
        # Schema is JSON-encoded into a string for prompt_metadata's value type.
        schema = json.loads(metadata["json_schema"])
        assert schema["type"] == "object"

    async def test_system_prompt_probe_sends_system_role(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.SYSTEM_PROMPT},
        )

        message: Message = target.send_prompt_async.await_args.kwargs["message"]
        roles = [piece.role for piece in message.message_pieces]
        assert "system" in roles

    async def test_multi_message_pieces_probe_sends_two_pieces(self) -> None:
        target = MockPromptTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_MESSAGE_PIECES},
        )

        message: Message = target.send_prompt_async.await_args.kwargs["message"]
        assert len(message.message_pieces) == 2

    async def test_probes_run_under_permissive_configuration(self) -> None:
        """
        Even when the target declares no boolean capabilities, the probe should
        still execute because the configuration is temporarily permissive.
        """
        target = MockPromptTarget()
        target._configuration = TargetConfiguration(capabilities=TargetCapabilities())
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_MESSAGE_PIECES},
        )

        # Probe was actually invoked.
        assert target.send_prompt_async.await_count >= 1
        assert CapabilityName.MULTI_MESSAGE_PIECES in result


@pytest.mark.usefixtures("patch_central_database")
class TestQueryTargetCapabilitiesIsolatedTarget:
    """Tests using a bare PromptTarget subclass (no PromptChatTarget extras)."""

    async def test_with_minimal_target_subclass(self) -> None:
        class _MinimalTarget(PromptTarget):
            async def _send_prompt_to_target_async(
                self, *, normalized_conversation: list[Message]
            ) -> list[Message]:
                return _ok_response()

        target = _MinimalTarget()
        target.send_prompt_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        result = await query_target_capabilities_async(target=target)

        for capability in _CAPABILITY_PROBES:
            assert capability in result


# ---------------------------------------------------------------------------
# Modality verification tests
# ---------------------------------------------------------------------------


def _set_input_modalities(
    *,
    target: MockPromptTarget,
    modalities: set[frozenset[PromptDataType]],
) -> None:
    target._configuration = TargetConfiguration(
        capabilities=TargetCapabilities(
            input_modalities=frozenset(modalities),
        ),
    )


@pytest.fixture
def image_asset(tmp_path: Path) -> str:
    """Create a tiny placeholder file usable as an image_path asset."""
    asset = tmp_path / "test_image.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(asset)


@pytest.mark.usefixtures("patch_central_database")
class TestCreateTestMessage:
    def test_text_only(self) -> None:
        msg = _create_test_message(modalities=frozenset({"text"}), test_assets={})
        assert len(msg.message_pieces) == 1
        assert msg.message_pieces[0].original_value_data_type == "text"

    def test_multimodal_uses_assets(self, image_asset: str) -> None:
        msg = _create_test_message(
            modalities=frozenset({"text", "image_path"}),
            test_assets={"image_path": image_asset},
        )
        types = {piece.original_value_data_type for piece in msg.message_pieces}
        assert types == {"text", "image_path"}

        # All pieces share the same conversation_id (Message.validate requires it).
        conv_ids = {piece.conversation_id for piece in msg.message_pieces}
        assert len(conv_ids) == 1

    def test_missing_asset_file_raises_filenotfound(self, tmp_path: Path) -> None:
        missing_path = str(tmp_path / "does_not_exist.png")
        with pytest.raises(FileNotFoundError):
            _create_test_message(
                modalities=frozenset({"image_path"}),
                test_assets={"image_path": missing_path},
            )

    def test_unconfigured_modality_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="No test asset configured"):
            _create_test_message(
                modalities=frozenset({"image_path"}),
                test_assets={},
            )


@pytest.mark.usefixtures("patch_central_database")
class TestVerifyTargetModalitiesAsync:
    async def test_all_combinations_supported(self) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await verify_target_modalities_async(target=target)

        assert frozenset({"text"}) in result

    async def test_exception_excludes_combination(self) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target.send_prompt_async = AsyncMock(side_effect=Exception("nope"))

        result = await verify_target_modalities_async(target=target)

        assert result == set()

    async def test_error_response_excludes_combination(self) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target.send_prompt_async = AsyncMock(return_value=_error_response())

        result = await verify_target_modalities_async(target=target)

        assert result == set()

    async def test_partial_support_via_selective_failure(self, image_asset: str) -> None:
        target = MockPromptTarget()
        _set_input_modalities(
            target=target,
            modalities={frozenset({"text"}), frozenset({"text", "image_path"})},
        )

        async def selective_send(*, message: Message) -> list[Message]:
            types = {p.original_value_data_type for p in message.message_pieces}
            if "image_path" in types:
                raise Exception("image not supported")
            return _ok_response()

        target.send_prompt_async = selective_send  # type: ignore[method-assign]

        result = await verify_target_modalities_async(
            target=target,
            test_assets={"image_path": image_asset},
        )

        assert frozenset({"text"}) in result
        assert frozenset({"text", "image_path"}) not in result

    async def test_explicit_test_modalities_overrides_declared(self, image_asset: str) -> None:
        target = MockPromptTarget()
        # Declared as text-only, but caller asks us to probe text+image too.
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        result = await verify_target_modalities_async(
            target=target,
            test_modalities={frozenset({"text"}), frozenset({"text", "image_path"})},
            test_assets={"image_path": image_asset},
        )

        assert frozenset({"text"}) in result
        assert frozenset({"text", "image_path"}) in result

    async def test_combination_skipped_when_asset_missing(self, tmp_path: Path) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text", "image_path"})})
        target.send_prompt_async = AsyncMock(return_value=_ok_response())

        # No assets provided — image_path combinations are skipped, not probed.
        result = await verify_target_modalities_async(target=target)

        assert result == set()
        assert target.send_prompt_async.await_count == 0
