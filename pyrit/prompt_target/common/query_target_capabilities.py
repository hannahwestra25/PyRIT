# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Runtime capability and modality discovery for prompt targets.

This module exposes two complementary probes:

* :func:`query_target_capabilities_async` probes the boolean capability flags
  defined on :class:`TargetCapabilities` (e.g. ``supports_system_prompt``,
  ``supports_multi_message_pieces``). For each capability that has a probe
  defined, a minimal request is sent to the target. If the request succeeds,
  the capability is included in the returned set. Capabilities without a
  registered probe fall back to whatever the target declares via its
  :class:`TargetConfiguration`.
* :func:`verify_target_modalities_async` probes which input modality
  combinations a target actually supports by sending a minimal test request
  for each combination declared in ``TargetCapabilities.input_modalities``.
"""

import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace

from pyrit.models import Message, MessagePiece, PromptDataType
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

logger = logging.getLogger(__name__)


_CapabilityProbe = Callable[[PromptTarget], Awaitable[bool]]


_PERMISSIVE_POLICY = CapabilityHandlingPolicy(
    behaviors={
        CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
        CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
    }
)


@contextmanager
def _permissive_configuration(*, target: PromptTarget) -> Iterator[None]:
    """
    Temporarily replace ``target``'s configuration with one that declares every
    boolean capability as natively supported.

    This bypasses :meth:`PromptTarget._validate_request`, which would otherwise
    short-circuit probes for capabilities the target declares as unsupported
    before any API call is made. The original configuration is restored on exit.

    Args:
        target (PromptTarget): The target whose configuration is temporarily replaced.

    Yields:
        None: Control returns to the ``with`` block while the permissive
        configuration is in effect.
    """
    original = target.configuration
    permissive_caps = replace(
        original.capabilities,
        supports_multi_turn=True,
        supports_multi_message_pieces=True,
        supports_json_schema=True,
        supports_json_output=True,
        supports_editable_history=True,
        supports_system_prompt=True,
    )
    target._configuration = TargetConfiguration(
        capabilities=permissive_caps,
        policy=_PERMISSIVE_POLICY,
    )
    try:
        yield
    finally:
        target._configuration = original


def _new_conversation_id() -> str:
    """
    Generate a unique conversation id for a single capability probe.

    Returns:
        str: A conversation id of the form ``"capability-probe-<uuid>"``.
    """
    return f"capability-probe-{uuid.uuid4()}"


def _user_text_piece(*, value: str, conversation_id: str) -> MessagePiece:
    """
    Build a single user-role text :class:`MessagePiece` for use in a probe.

    Args:
        value (str): The text payload to send.
        conversation_id (str): The conversation id to attach to the piece.

    Returns:
        MessagePiece: A user-role text piece bound to ``conversation_id``.
    """
    return MessagePiece(
        role="user",
        original_value=value,
        original_value_data_type="text",
        conversation_id=conversation_id,
    )


async def _send_and_check_async(*, target: PromptTarget, message: Message) -> bool:
    """
    Send ``message`` and report whether the call succeeded cleanly.

    Args:
        target (PromptTarget): The target to send the probe message to.
        message (Message): The probe message to send.

    Returns:
        bool: ``True`` iff the call returned without raising and every response
        piece reported ``response_error == "none"``; ``False`` otherwise.
    """
    try:
        responses = await target.send_prompt_async(message=message)
    except Exception as exc:
        logger.info("Capability probe failed: %s", exc)
        return False

    for response in responses:
        for piece in response.message_pieces:
            if piece.response_error != "none":
                logger.info("Capability probe returned error response: %s", piece.converted_value)
                return False
    return True


async def _probe_system_prompt_async(target: PromptTarget) -> bool:
    """
    Probe whether ``target`` accepts a system message alongside a user message.

    Args:
        target (PromptTarget): The target to probe.

    Returns:
        bool: ``True`` if the system + user request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    system_piece = MessagePiece(
        role="system",
        original_value="You are a helpful assistant.",
        original_value_data_type="text",
        conversation_id=conversation_id,
    )
    user_piece = _user_text_piece(value="hi", conversation_id=conversation_id)
    return await _send_and_check_async(target=target, message=Message([system_piece, user_piece]))


async def _probe_multi_message_pieces_async(target: PromptTarget) -> bool:
    """
    Probe whether ``target`` accepts a single message containing multiple pieces.

    Args:
        target (PromptTarget): The target to probe.

    Returns:
        bool: ``True`` if the multi-piece request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    pieces = [
        _user_text_piece(value="part one", conversation_id=conversation_id),
        _user_text_piece(value="part two", conversation_id=conversation_id),
    ]
    return await _send_and_check_async(target=target, message=Message(pieces))


async def _probe_multi_turn_async(target: PromptTarget) -> bool:
    """
    Probe whether ``target`` accepts two sequential messages on the same conversation.

    Args:
        target (PromptTarget): The target to probe.

    Returns:
        bool: ``True`` if both turns succeeded; ``False`` if either turn failed.
    """
    conversation_id = _new_conversation_id()
    first = _user_text_piece(value="hello", conversation_id=conversation_id)
    if not await _send_and_check_async(target=target, message=Message([first])):
        return False
    second = _user_text_piece(value="and again", conversation_id=conversation_id)
    return await _send_and_check_async(target=target, message=Message([second]))


async def _probe_json_output_async(target: PromptTarget) -> bool:
    """
    Probe whether ``target`` accepts a request asking for JSON-mode output.

    Args:
        target (PromptTarget): The target to probe.

    Returns:
        bool: ``True`` if the JSON-mode request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    piece = MessagePiece(
        role="user",
        original_value='Respond with a JSON object: {"ok": true}.',
        original_value_data_type="text",
        conversation_id=conversation_id,
        prompt_metadata={"response_format": "json"},
    )
    return await _send_and_check_async(target=target, message=Message([piece]))


async def _probe_json_schema_async(target: PromptTarget) -> bool:
    """
    Probe whether ``target`` accepts a request constrained by a JSON schema.

    Args:
        target (PromptTarget): The target to probe.

    Returns:
        bool: ``True`` if the schema-constrained request succeeded; ``False`` otherwise.
    """
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    conversation_id = _new_conversation_id()
    piece = MessagePiece(
        role="user",
        original_value='Respond with a JSON object matching the schema: {"ok": true}.',
        original_value_data_type="text",
        conversation_id=conversation_id,
        prompt_metadata={
            "response_format": "json",
            "json_schema": json.dumps(schema),
        },
    )
    return await _send_and_check_async(target=target, message=Message([piece]))


# Registry of capabilities that can be verified via a live API call.
# Capabilities not present here fall back to the target's declared support.
_CAPABILITY_PROBES: dict[CapabilityName, _CapabilityProbe] = {
    CapabilityName.SYSTEM_PROMPT: _probe_system_prompt_async,
    CapabilityName.MULTI_MESSAGE_PIECES: _probe_multi_message_pieces_async,
    CapabilityName.MULTI_TURN: _probe_multi_turn_async,
    CapabilityName.JSON_OUTPUT: _probe_json_output_async,
    CapabilityName.JSON_SCHEMA: _probe_json_schema_async,
}


async def query_target_capabilities_async(
    *,
    target: PromptTarget,
    capabilities: Iterable[CapabilityName] | None = None,
) -> set[CapabilityName]:
    """
    Probe ``target`` to determine which capabilities it actually supports.

    For each requested capability that has a registered probe, a minimal
    request is sent to the target. The capability is treated as supported
    only if the call returns successfully with no error response. For
    capabilities without a registered probe, the target's declared support
    (``target.configuration.includes(...)``) is used as a fallback.

    During probing, the target's configuration is temporarily replaced with
    one that declares every boolean capability as supported, so that
    :meth:`PromptTarget._validate_request` does not short-circuit probes for
    capabilities the target declares as unsupported. The original
    configuration is restored before this function returns.

    Args:
        target (PromptTarget): The target to probe.
        capabilities (Iterable[CapabilityName] | None): Capabilities to check.
            Defaults to every member of :class:`CapabilityName`.

    Returns:
        set[CapabilityName]: The capabilities verified to work against the target.
    """
    capabilities_to_check: Iterable[CapabilityName] = capabilities if capabilities is not None else CapabilityName

    verified: set[CapabilityName] = set()
    with _permissive_configuration(target=target):
        for capability in capabilities_to_check:
            probe = _CAPABILITY_PROBES.get(capability)
            if probe is None:
                # No live probe; fall back to whatever the (original) configuration declared.
                # We're inside the permissive override, so consult the saved configuration directly.
                continue

            try:
                if await probe(target):
                    verified.add(capability)
            except Exception as exc:
                logger.info("Probe for %s raised: %s", capability.value, exc)

    # Add capabilities without a probe based on the original (now-restored) declared support.
    for capability in capabilities_to_check:
        if capability not in _CAPABILITY_PROBES and target.configuration.includes(capability=capability):
            verified.add(capability)

    return verified


# ---------------------------------------------------------------------------
# Modality verification
# ---------------------------------------------------------------------------


# Default mapping of non-text modalities to test asset paths. Callers can
# override via the ``test_assets`` parameter of
# :func:`verify_target_modalities_async`. Modalities whose assets do not
# exist on disk are skipped (logged and excluded from the result).
DEFAULT_TEST_ASSETS: dict[PromptDataType, str] = {}


async def verify_target_modalities_async(
    *,
    target: PromptTarget,
    test_modalities: set[frozenset[PromptDataType]] | None = None,
    test_assets: dict[PromptDataType, str] | None = None,
) -> set[frozenset[PromptDataType]]:
    """
    Probe ``target`` to determine which input modality combinations it supports.

    Each combination is exercised with a minimal request built by
    :func:`_create_test_message`. A combination is considered supported only
    if the request returns successfully with no error response.

    Args:
        target (PromptTarget): The target to probe.
        test_modalities (set[frozenset[PromptDataType]] | None): Specific
            modality combinations to test. Defaults to the combinations
            declared in ``target.capabilities.input_modalities``.
        test_assets (dict[PromptDataType, str] | None): Mapping from
            non-text modality to a file path used as the probe payload.
            Defaults to :data:`DEFAULT_TEST_ASSETS`. Combinations whose
            non-text assets are missing on disk are skipped.

    Returns:
        set[frozenset[PromptDataType]]: The modality combinations verified
        to work against the target.
    """
    if test_modalities is None:
        declared = target.capabilities.input_modalities
        test_modalities = set(declared)

    assets = test_assets if test_assets is not None else DEFAULT_TEST_ASSETS

    verified: set[frozenset[PromptDataType]] = set()
    for combination in test_modalities:
        try:
            message = _create_test_message(modalities=combination, test_assets=assets)
        except FileNotFoundError as exc:
            logger.info("Skipping modality %s: %s", combination, exc)
            continue
        except ValueError as exc:
            logger.info("Skipping modality %s: %s", combination, exc)
            continue

        if await _test_modality_combination_async(target=target, message=message):
            verified.add(combination)

    return verified


async def _test_modality_combination_async(*, target: PromptTarget, message: Message) -> bool:
    """
    Send a modality probe ``message`` and report whether the call succeeded cleanly.

    Args:
        target (PromptTarget): The target to send the probe message to.
        message (Message): The probe message exercising a specific modality combination.

    Returns:
        bool: ``True`` iff the call returned without raising and every response
        piece reported ``response_error == "none"``; ``False`` otherwise.
    """
    try:
        responses = await target.send_prompt_async(message=message)
    except Exception as exc:
        logger.info("Modality probe failed: %s", exc)
        return False

    for response in responses:
        for piece in response.message_pieces:
            if piece.response_error != "none":
                logger.info("Modality probe returned error response: %s", piece.converted_value)
                return False
    return True


def _create_test_message(
    *,
    modalities: frozenset[PromptDataType],
    test_assets: dict[PromptDataType, str],
) -> Message:
    """
    Build a minimal :class:`Message` that exercises ``modalities``.

    Args:
        modalities (frozenset[PromptDataType]): The modalities to include.
        test_assets (dict[PromptDataType, str]): Mapping from non-text
            modality to a file path used for the probe.

    Returns:
        Message: A message containing one piece per modality.

    Raises:
        FileNotFoundError: If a configured asset path does not exist.
        ValueError: If a non-text modality has no configured asset, or if
            no pieces could be constructed.
    """
    conversation_id = f"modality-probe-{uuid.uuid4()}"
    pieces: list[MessagePiece] = []

    for modality in modalities:
        if modality == "text":
            pieces.append(
                MessagePiece(
                    role="user",
                    original_value="test",
                    original_value_data_type="text",
                    conversation_id=conversation_id,
                )
            )
            continue

        asset_path = test_assets.get(modality)
        if asset_path is None:
            raise ValueError(f"No test asset configured for modality '{modality}'.")
        if not os.path.isfile(asset_path):
            raise FileNotFoundError(f"Test asset for modality '{modality}' not found at: {asset_path}")

        pieces.append(
            MessagePiece(
                role="user",
                original_value=asset_path,
                original_value_data_type=modality,
                conversation_id=conversation_id,
            )
        )

    if not pieces:
        raise ValueError(f"Could not create test message for modalities: {modalities}")

    return Message(pieces)
