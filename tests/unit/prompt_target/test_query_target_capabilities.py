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


class _RealValidationTarget(PromptTarget):
    """
    Bare ``PromptTarget`` subclass that does NOT override ``_validate_request``.

    Tests that need to verify ``_permissive_configuration`` actually bypasses
    the validation guard use this instead of ``MockPromptTarget`` (which
    no-ops ``_validate_request``).
    """

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(),
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        return _ok_response()


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
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        result = await query_target_capabilities_async(target=target)

        # Every capability with a probe should be in the result.
        for capability in _CAPABILITY_PROBES:
            assert capability in result

    async def test_excludes_capabilities_when_probe_fails(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(side_effect=Exception("nope"))  # type: ignore[method-assign]

        result = await query_target_capabilities_async(target=target)

        for capability in _CAPABILITY_PROBES:
            assert capability not in result

    async def test_excludes_capabilities_when_response_has_error(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_error_response())  # type: ignore[method-assign]

        result = await query_target_capabilities_async(target=target)

        for capability in _CAPABILITY_PROBES:
            assert capability not in result

    async def test_filters_by_requested_capabilities(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        requested = {CapabilityName.SYSTEM_PROMPT, CapabilityName.MULTI_TURN}
        result = await query_target_capabilities_async(target=target, capabilities=requested)

        assert result == requested

    async def test_capability_without_probe_falls_back_to_declared_support(self) -> None:
        target = MockPromptTarget()
        # Override the configuration so editable_history is declared as supported.
        target._configuration = TargetConfiguration(
            capabilities=TargetCapabilities(supports_editable_history=True),
        )
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.EDITABLE_HISTORY},
        )

        assert result == {CapabilityName.EDITABLE_HISTORY}

    async def test_capability_without_probe_excluded_when_not_declared(self) -> None:
        target = MockPromptTarget()
        # Override to a configuration that does NOT declare editable_history.
        target._configuration = TargetConfiguration(capabilities=TargetCapabilities())
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.EDITABLE_HISTORY},
        )

        assert result == set()

    async def test_capability_without_probe_excluded_when_only_adapted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        ADAPT in the policy must NOT count as native support for the fallback.

        Today every adaptable capability also has a probe, so this scenario only
        arises if a future capability is declared adaptable without a probe.
        We simulate that by removing SYSTEM_PROMPT from the registry and
        configuring the target with ``ADAPT`` for it but no native support.
        """
        from pyrit.prompt_target.common import query_target_capabilities as qtc
        from pyrit.prompt_target.common.target_capabilities import (
            CapabilityHandlingPolicy,
            UnsupportedCapabilityBehavior,
        )

        patched_probes = {k: v for k, v in qtc._CAPABILITY_PROBES.items() if k is not CapabilityName.SYSTEM_PROMPT}
        monkeypatch.setattr(qtc, "_CAPABILITY_PROBES", patched_probes)

        target = MockPromptTarget()
        target._configuration = TargetConfiguration(
            capabilities=TargetCapabilities(),  # no native SYSTEM_PROMPT
            policy=CapabilityHandlingPolicy(
                behaviors={
                    CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
                    CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                }
            ),
        )

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.SYSTEM_PROMPT},
        )

        assert result == set()

    async def test_restores_configuration_after_probing(self) -> None:
        target = MockPromptTarget()
        original = target.configuration
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        await query_target_capabilities_async(target=target)

        assert target.configuration is original

    async def test_multi_turn_probe_sends_history_on_second_call(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_TURN},
        )

        # Multi-turn probe sends two requests on the same conversation_id, and
        # seeds memory between them so the second call carries real history.
        calls = target._send_prompt_to_target_async.await_args_list
        assert len(calls) == 2

        first_conv = calls[0].kwargs["normalized_conversation"]
        second_conv = calls[1].kwargs["normalized_conversation"]

        first_conv_id = first_conv[-1].message_pieces[0].conversation_id
        second_conv_id = second_conv[-1].message_pieces[0].conversation_id
        assert first_conv_id == second_conv_id

        # First call is a single-turn user message; the second call must include
        # the seeded user + assistant history followed by the new user turn.
        assert len(first_conv) == 1
        assert len(second_conv) >= 3
        roles = [msg.message_pieces[0]._role for msg in second_conv]
        assert roles[-3:] == ["user", "assistant", "user"]

    async def test_multi_turn_probe_short_circuits_on_first_failure(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(side_effect=Exception("first call fails"))  # type: ignore[method-assign]

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_TURN},
        )

        assert result == set()
        # _send_and_check_async retries once on exception, so the failing
        # first turn is attempted twice; the second turn is never reached.
        assert target._send_prompt_to_target_async.await_count == 2

    async def test_json_schema_probe_sends_schema_in_metadata(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.JSON_SCHEMA},
        )

        normalized: list[Message] = target._send_prompt_to_target_async.await_args.kwargs["normalized_conversation"]
        metadata = normalized[-1].message_pieces[0].prompt_metadata
        assert metadata is not None
        assert metadata["response_format"] == "json"
        # Schema is JSON-encoded into a string for prompt_metadata's value type.
        schema = json.loads(metadata["json_schema"])
        assert schema["type"] == "object"

    async def test_system_prompt_probe_installs_system_message_and_sends_user(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.SYSTEM_PROMPT},
        )

        # The probe writes a system message directly to memory (bypassing
        # PromptTarget.set_system_prompt, which subclasses can override) and
        # then sends a user-role message. Message.validate forbids mixed
        # roles in a single Message, so the system and user turns are
        # separate. Verify the system message is in memory and the wire
        # payload contains the system + user history.
        normalized: list[Message] = target._send_prompt_to_target_async.await_args.kwargs["normalized_conversation"]
        roles_sent = [piece._role for msg in normalized for piece in msg.message_pieces]
        assert "system" in roles_sent
        assert roles_sent[-1] == "user"
        # The last sent Message itself should be user-only.
        assert [piece._role for piece in normalized[-1].message_pieces] == ["user"]

    async def test_multi_message_pieces_probe_sends_two_pieces(self) -> None:
        target = MockPromptTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_MESSAGE_PIECES},
        )

        normalized: list[Message] = target._send_prompt_to_target_async.await_args.kwargs["normalized_conversation"]
        assert len(normalized[-1].message_pieces) == 2

    async def test_probes_run_under_permissive_configuration(self) -> None:
        """
        Even when the target declares no boolean capabilities, the probe should
        still execute because the configuration is temporarily permissive.

        Uses ``_RealValidationTarget`` so that ``_validate_request`` actually
        runs and would reject the multi-piece probe were the override absent.
        """
        target = _RealValidationTarget()
        send_mock = AsyncMock(return_value=_ok_response())
        target._send_prompt_to_target_async = send_mock  # type: ignore[method-assign]

        result = await query_target_capabilities_async(
            target=target,
            capabilities={CapabilityName.MULTI_MESSAGE_PIECES},
        )

        # Probe was actually invoked through the full send_prompt_async pipeline,
        # which means _validate_request ran and was satisfied by the permissive
        # override (the bare target declares no capabilities natively).
        assert send_mock.await_count >= 1
        assert CapabilityName.MULTI_MESSAGE_PIECES in result


@pytest.mark.usefixtures("patch_central_database")
class TestQueryTargetCapabilitiesIsolatedTarget:
    """Tests using a bare PromptTarget subclass (no PromptChatTarget extras)."""

    async def test_with_minimal_target_subclass(self) -> None:
        class _MinimalTarget(PromptTarget):
            async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
                return _ok_response()

        target = _MinimalTarget()
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

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
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        result = await verify_target_modalities_async(target=target)

        assert frozenset({"text"}) in result

    async def test_exception_excludes_combination(self) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target._send_prompt_to_target_async = AsyncMock(side_effect=Exception("nope"))  # type: ignore[method-assign]

        result = await verify_target_modalities_async(target=target)

        assert result == set()

    async def test_error_response_excludes_combination(self) -> None:
        target = MockPromptTarget()
        _set_input_modalities(target=target, modalities={frozenset({"text"})})
        target._send_prompt_to_target_async = AsyncMock(return_value=_error_response())  # type: ignore[method-assign]

        result = await verify_target_modalities_async(target=target)

        assert result == set()

    async def test_partial_support_via_selective_failure(self, image_asset: str) -> None:
        target = MockPromptTarget()
        _set_input_modalities(
            target=target,
            modalities={frozenset({"text"}), frozenset({"text", "image_path"})},
        )

        async def selective_send(*, normalized_conversation: list[Message]) -> list[Message]:
            message = normalized_conversation[-1]
            types = {p.original_value_data_type for p in message.message_pieces}
            if "image_path" in types:
                raise Exception("image not supported")
            return _ok_response()

        target._send_prompt_to_target_async = selective_send  # type: ignore[method-assign]

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
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

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
        target._send_prompt_to_target_async = AsyncMock(return_value=_ok_response())  # type: ignore[method-assign]

        # No assets provided — image_path combinations are skipped, not probed.
        result = await verify_target_modalities_async(target=target)

        assert result == set()
        assert target._send_prompt_to_target_async.await_count == 0

    async def test_explicit_test_modalities_runs_under_permissive_configuration(self, image_asset: str) -> None:
        """
        Probing a modality combination the target does NOT declare must still
        succeed. Uses ``_RealValidationTarget`` so ``_validate_request`` runs
        and would reject the multi-piece, non-text payload were the
        permissive override absent.
        """
        target = _RealValidationTarget()
        send_mock = AsyncMock(return_value=_ok_response())
        target._send_prompt_to_target_async = send_mock  # type: ignore[method-assign]

        result = await verify_target_modalities_async(
            target=target,
            test_modalities={frozenset({"text", "image_path"})},
            test_assets={"image_path": image_asset},
        )

        assert send_mock.await_count == 1
        assert frozenset({"text", "image_path"}) in result
