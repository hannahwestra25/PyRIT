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

.. note::
   Output modality probing is intentionally not provided. Unlike inputs,
   output modality is largely a property of the endpoint type (chat models
   return text, image models return images, TTS endpoints return audio)
   rather than something the caller controls per request, and there is no
   PyRIT-level ``response_format=image`` style hint to assert against.
   Eliciting non-text output reliably depends on prompt phrasing, costs
   real compute per probe, and is prone to false negatives from safety
   filters. Trust ``target.capabilities.output_modalities`` as declared.

.. warning::
   These probes only verify that a request was *accepted* (the call returned
   without raising and the response had no error). They cannot detect a
   target that silently ignores a feature. For example, an endpoint that
   accepts a ``system`` role but discards it, or that accepts a
   ``response_format="json"`` hint but returns prose, will be reported as
   supporting those capabilities. Treat the returned sets as an upper bound
   on actual support and validate response content out of band when the
   distinction matters (e.g. parse JSON responses, assert that the model
   honored the system prompt).
"""

import asyncio
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
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

logger = logging.getLogger(__name__)

# Per-call timeout (seconds) applied to every probe request. Override per-call via
# the ``per_probe_timeout_s`` parameter on the public functions.
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 30.0

# Marker stamped onto every MessagePiece this module writes to memory. Consumers
# that aggregate or display memory rows can filter probe-written rows by checking
# ``piece.prompt_metadata.get("capability_probe") == "1"``. Memory does not yet
# expose a delete-by-conversation-id API, so tagging is the cleanup mechanism.
PROBE_METADATA_KEY: str = "capability_probe"
PROBE_METADATA_VALUE: str = "1"

_CapabilityProbe = Callable[[PromptTarget, float, int], Awaitable[bool]]


_PROBE_POLICY = CapabilityHandlingPolicy(
    behaviors={
        CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
        CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
    }
)


# Every text probe sends a text-only payload. Permissive overrides therefore
# always include this combination so that ``_validate_request``'s per-piece
# data-type check does not reject text probes against text-less targets.
_TEXT_MODALITY: frozenset[frozenset[PromptDataType]] = frozenset({frozenset({"text"})})


@contextmanager
def _permissive_configuration(
    *,
    target: PromptTarget,
    extra_input_modalities: Iterable[frozenset[PromptDataType]] | None = None,
) -> Iterator[None]:
    """
    Temporarily replace ``target``'s configuration with one that declares every
    boolean capability as natively supported.

    This bypasses :meth:`PromptTarget._validate_request`, which would otherwise
    short-circuit probes for capabilities the target declares as unsupported
    before any API call is made. The original configuration is restored on exit.

    Args:
        target (PromptTarget): The target whose configuration is temporarily replaced.
        extra_input_modalities (Iterable[frozenset[PromptDataType]] | None):
            Additional modality combinations to include in ``input_modalities``
            during the override. Used by modality probes so that
            ``_validate_request``'s per-piece data-type check does not reject
            combinations the caller asked us to test but the target does not
            yet declare. Defaults to None.

    Yields:
        None: Control returns to the ``with`` block while the permissive
        configuration is in effect.
    """
    original = target.configuration
    merged_modalities = original.capabilities.input_modalities | _TEXT_MODALITY
    if extra_input_modalities is not None:
        merged_modalities = frozenset(merged_modalities | frozenset(extra_input_modalities))
    permissive_caps = replace(
        original.capabilities,
        supports_multi_turn=True,
        supports_multi_message_pieces=True,
        supports_json_schema=True,
        supports_json_output=True,
        supports_editable_history=True,
        supports_system_prompt=True,
        input_modalities=merged_modalities,
    )
    target._configuration = TargetConfiguration(
        capabilities=permissive_caps,
        policy=_PROBE_POLICY,
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


def _probe_metadata(extra: dict[str, str | int] | None = None) -> dict[str, str | int]:
    """Return a fresh ``prompt_metadata`` dict tagged as a capability probe."""
    metadata: dict[str, str | int] = {PROBE_METADATA_KEY: PROBE_METADATA_VALUE}
    if extra:
        metadata.update(extra)
    return metadata


def _user_text_piece(*, value: str, conversation_id: str) -> MessagePiece:
    """
    Build a single user-role text :class:`MessagePiece` for use in a probe.

    The piece's ``prompt_metadata`` is tagged with :data:`PROBE_METADATA_KEY`
    so that consumers aggregating memory can filter out probe-written rows.

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
        prompt_metadata=_probe_metadata(),
    )


async def _send_and_check_async(
    *,
    target: PromptTarget,
    message: Message,
    timeout_s: float,
    retries: int = 1,
    label: str = "Capability probe",
) -> bool:
    """
    Send ``message`` and report whether the call succeeded cleanly.

    Each attempt is bounded by ``timeout_s``. Exceptions (network errors,
    timeouts, validation failures) trigger up to ``retries`` retries before
    the probe is declared failed; an explicit error response from the target
    is treated as deterministic and never retried.

    Args:
        target (PromptTarget): The target to send the probe message to.
        message (Message): The probe message to send.
        timeout_s (float): Per-attempt timeout in seconds.
        retries (int): Number of additional attempts after the first failure.
            Only exceptions are retried; a non-error response is final.
            Defaults to 1.
        label (str): Short label used in log messages. Defaults to
            ``"Capability probe"``.

    Returns:
        bool: ``True`` iff the call returned without raising and every response
        piece reported ``response_error == "none"``; ``False`` otherwise.
    """
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            responses = await asyncio.wait_for(target.send_prompt_async(message=message), timeout=timeout_s)
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"timed out after {timeout_s}s")
            logger.info("%s timed out (attempt %d/%d)", label, attempt + 1, attempts)
            continue
        except Exception as exc:
            last_exc = exc
            logger.info("%s failed (attempt %d/%d): %s", label, attempt + 1, attempts, exc)
            continue

        for response in responses:
            for piece in response.message_pieces:
                if piece.response_error != "none":
                    logger.info("%s returned error response: %s", label, piece.converted_value)
                    return False
        return True

    logger.info("%s exhausted %d attempt(s); last error: %s", label, attempts, last_exc)
    return False


async def _probe_system_prompt_async(target: PromptTarget, timeout_s: float, retries: int = 1) -> bool:
    """
    Probe whether ``target`` accepts a system prompt followed by a user message.

    Writes a system-role :class:`MessagePiece` directly to ``target._memory``
    rather than calling :meth:`PromptTarget.set_system_prompt`. ``set_system_prompt``
    can be overridden by subclasses (e.g. mocks) to do nothing or to perform
    extra work, which would mask whether the underlying API actually accepts a
    system message. A direct memory write guarantees the probe sees the same
    multi-piece, system-then-user payload the target's wire layer would see
    via the standard pipeline.

    Args:
        target (PromptTarget): The target to probe.
        timeout_s (float): Per-attempt timeout in seconds.

    Returns:
        bool: ``True`` if the system + user request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    system_piece = MessagePiece(
        role="system",
        original_value="You are a helpful assistant.",
        original_value_data_type="text",
        conversation_id=conversation_id,
        prompt_metadata=_probe_metadata(),
    )
    try:
        target._memory.add_message_to_memory(request=Message([system_piece]))
    except Exception as exc:
        logger.info("System-prompt probe could not seed system message: %s", exc)
        return False
    user_piece = _user_text_piece(value="hi", conversation_id=conversation_id)
    return await _send_and_check_async(
        target=target,
        message=Message([user_piece]),
        timeout_s=timeout_s,
        retries=retries,
        label="System-prompt probe",
    )


async def _probe_multi_message_pieces_async(target: PromptTarget, timeout_s: float, retries: int = 1) -> bool:
    """
    Probe whether ``target`` accepts a single message containing multiple pieces.

    Args:
        target (PromptTarget): The target to probe.
        timeout_s (float): Per-attempt timeout in seconds.

    Returns:
        bool: ``True`` if the multi-piece request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    pieces = [
        _user_text_piece(value="part one", conversation_id=conversation_id),
        _user_text_piece(value="part two", conversation_id=conversation_id),
    ]
    return await _send_and_check_async(
        target=target,
        message=Message(pieces),
        timeout_s=timeout_s,
        retries=retries,
        label="Multi-message-pieces probe",
    )


async def _probe_multi_turn_async(target: PromptTarget, timeout_s: float, retries: int = 1) -> bool:
    """
    Probe whether ``target`` accepts a request that includes prior conversation history.

    ``PromptTarget.send_prompt_async`` reads conversation history from memory but
    does not write to it (persistence normally happens in the orchestrator
    layer). To exercise true multi-turn behavior, this probe:

    1. Sends an initial user message.
    2. Persists that user message and a synthetic assistant reply directly to
       the target's memory under the same ``conversation_id``.
    3. Sends a second user message; ``send_prompt_async`` then fetches the
       2-message history and the target receives a real 3-message
       multi-turn payload.

    The synthetic assistant reply's content is irrelevant — we are testing
    whether the target's API accepts a multi-turn payload, not whether the
    model recalls anything.

    Args:
        target (PromptTarget): The target to probe.
        timeout_s (float): Per-attempt timeout in seconds.

    Returns:
        bool: ``True`` if both turns succeeded; ``False`` if either turn failed.
    """
    conversation_id = _new_conversation_id()
    first = _user_text_piece(value="My favorite color is blue.", conversation_id=conversation_id)
    if not await _send_and_check_async(
        target=target, message=Message([first]), timeout_s=timeout_s, retries=retries, label="Multi-turn probe (turn 1)"
    ):
        return False

    # Seed memory so the second send sees real prior history.
    target._memory.add_message_to_memory(request=Message([first]))
    assistant_reply = MessagePiece(
        role="assistant",
        original_value="Got it.",
        original_value_data_type="text",
        conversation_id=conversation_id,
        prompt_metadata=_probe_metadata(),
    ).to_message()
    target._memory.add_message_to_memory(request=assistant_reply)

    second = _user_text_piece(value="What did I just tell you?", conversation_id=conversation_id)
    return await _send_and_check_async(
        target=target, message=Message([second]), timeout_s=timeout_s, retries=retries, label="Multi-turn probe (turn 2)"
    )


async def _probe_json_output_async(target: PromptTarget, timeout_s: float, retries: int = 1) -> bool:
    """
    Probe whether ``target`` accepts a request asking for JSON-mode output.

    Args:
        target (PromptTarget): The target to probe.
        timeout_s (float): Per-attempt timeout in seconds.

    Returns:
        bool: ``True`` if the JSON-mode request succeeded; ``False`` otherwise.
    """
    conversation_id = _new_conversation_id()
    piece = MessagePiece(
        role="user",
        original_value='Respond with a JSON object: {"ok": true}.',
        original_value_data_type="text",
        conversation_id=conversation_id,
        prompt_metadata=_probe_metadata({"response_format": "json"}),
    )
    return await _send_and_check_async(
        target=target, message=Message([piece]), timeout_s=timeout_s, retries=retries, label="JSON-output probe"
    )


async def _probe_json_schema_async(target: PromptTarget, timeout_s: float, retries: int = 1) -> bool:
    """
    Probe whether ``target`` accepts a request constrained by a JSON schema.

    Args:
        target (PromptTarget): The target to probe.
        timeout_s (float): Per-attempt timeout in seconds.

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
        prompt_metadata=_probe_metadata(
            {
                "response_format": "json",
                "json_schema": json.dumps(schema),
            }
        ),
    )
    return await _send_and_check_async(
        target=target, message=Message([piece]), timeout_s=timeout_s, retries=retries, label="JSON-schema probe"
    )


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
    per_probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    retries: int = 1,
) -> set[CapabilityName]:
    """
    Probe ``target`` to determine which capabilities it actually supports.

    For each requested capability that has a registered probe, a minimal
    request is sent to the target. The capability is treated as supported
    only if the call returns successfully with no error response. For
    capabilities without a registered probe, the target's declared
    **native** support (``target.capabilities.includes(...)``) is used as
    a fallback. We deliberately do *not* consult
    ``target.configuration.includes(...)`` here, because that would also
    return ``True`` for capabilities the target lacks but PyRIT
    ``ADAPT``s via the :class:`CapabilityHandlingPolicy` — and adaptation
    is an emulation by PyRIT, not evidence that the target itself supports
    the capability.

    .. warning::
       "Supported" here means "the request was accepted", not "the feature
       was actually applied". A target that silently ignores a system
       prompt, ``response_format``, or schema directive will still be
       reported as supporting that capability. Validate response content
       out of band when correctness matters.

    .. warning::
       This function is **not safe to call concurrently** with other
       operations on the same ``target`` instance. It temporarily mutates
       ``target._configuration`` and writes probe rows to ``target._memory``;
       concurrent callers may observe the permissive configuration or
       interleaved memory rows. Probe-written memory rows are tagged with
       ``prompt_metadata["capability_probe"] == "1"`` so consumers can
       filter them; memory does not currently expose a delete-by-conversation
       API, so probe rows persist for the lifetime of the memory backend.

    During probing, the target's configuration is temporarily replaced with
    one that declares every boolean capability as supported, so that
    :meth:`PromptTarget._validate_request` does not short-circuit probes for
    capabilities the target declares as unsupported. The original
    configuration is restored before this function returns.

    Args:
        target (PromptTarget): The target to probe.
        capabilities (Iterable[CapabilityName] | None): Capabilities to check.
            Defaults to every member of :class:`CapabilityName`.
        per_probe_timeout_s (float): Per-attempt timeout (seconds) applied to
            each probe request. Defaults to
            :data:`DEFAULT_PROBE_TIMEOUT_SECONDS`.
        retries (int): Number of additional attempts after the first failure
            for each probe. Only exceptions/timeouts are retried; an explicit
            error response is final. Set to ``0`` to disable retries.
            Defaults to 1.

    Returns:
        set[CapabilityName]: The capabilities verified to work against the target.
    """
    capabilities_to_check: list[CapabilityName] = (
        list(capabilities) if capabilities is not None else list(CapabilityName)
    )

    verified: set[CapabilityName] = set()
    with _permissive_configuration(target=target):
        for capability in capabilities_to_check:
            probe = _CAPABILITY_PROBES.get(capability)
            if probe is None:
                # No live probe; fall back to whatever the (original) configuration declared.
                # We're inside the permissive override, so consult the saved configuration directly.
                continue

            try:
                if await probe(target, per_probe_timeout_s, retries):
                    verified.add(capability)
            except Exception as exc:
                logger.info("Probe for %s raised: %s", capability.value, exc)

    # Add capabilities without a probe based on the original (now-restored) NATIVE
    # support. Using target.capabilities.includes (native flags) rather than
    # target.configuration.includes (which also returns True for ADAPT'd capabilities)
    # keeps this function's contract honest: we report only what the target itself
    # supports, never what PyRIT emulates on top of it.
    for capability in capabilities_to_check:
        if capability not in _CAPABILITY_PROBES and target.capabilities.includes(capability=capability):
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
    per_probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    retries: int = 1,
) -> set[frozenset[PromptDataType]]:
    """
    Probe ``target`` to determine which input modality combinations it supports.

    Each combination is exercised with a minimal request built by
    :func:`_create_test_message`. A combination is considered supported only
    if the request returns successfully with no error response.

    During probing the target's configuration is temporarily replaced with
    one that declares every boolean capability as natively supported and
    that includes every probed modality combination in ``input_modalities``,
    so :meth:`PromptTarget._validate_request` does not short-circuit a probe
    before any API call is made. The original configuration is restored
    before this function returns.

    .. warning::
       "Supported" here means the target accepted the request. A target
       that accepts e.g. an ``image_path`` piece but ignores its content
       will still be reported as supporting that modality.

    .. warning::
       This function is **not safe to call concurrently** with other
       operations on the same ``target`` instance. It temporarily mutates
       ``target._configuration``.

    Args:
        target (PromptTarget): The target to probe.
        test_modalities (set[frozenset[PromptDataType]] | None): Specific
            modality combinations to test. Defaults to the combinations
            declared in ``target.capabilities.input_modalities``.
        test_assets (dict[PromptDataType, str] | None): Mapping from
            non-text modality to a file path used as the probe payload.
            Defaults to :data:`DEFAULT_TEST_ASSETS`. Combinations whose
            non-text assets are missing on disk are skipped.
        per_probe_timeout_s (float): Per-attempt timeout (seconds) applied to
            each probe request. Defaults to
            :data:`DEFAULT_PROBE_TIMEOUT_SECONDS`.
        retries (int): Number of additional attempts after the first failure
            for each probe. Only exceptions/timeouts are retried; an explicit
            error response is final. Set to ``0`` to disable retries.
            Defaults to 1.

    Returns:
        set[frozenset[PromptDataType]]: The modality combinations verified
        to work against the target.
    """
    if test_modalities is None:
        declared = target.capabilities.input_modalities
        test_modalities = set(declared)

    assets = test_assets if test_assets is not None else DEFAULT_TEST_ASSETS

    verified: set[frozenset[PromptDataType]] = set()
    with _permissive_configuration(target=target, extra_input_modalities=test_modalities):
        for combination in test_modalities:
            try:
                message = _create_test_message(modalities=combination, test_assets=assets)
            except FileNotFoundError as exc:
                logger.info("Skipping modality %s: %s", combination, exc)
                continue
            except ValueError as exc:
                logger.info("Skipping modality %s: %s", combination, exc)
                continue

            if await _send_and_check_async(
                target=target,
                message=message,
                timeout_s=per_probe_timeout_s,
                retries=retries,
                label=f"Modality probe {sorted(combination)}",
            ):
                verified.add(combination)

    return verified


async def verify_target_async(
    *,
    target: PromptTarget,
    per_probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    test_assets: dict[PromptDataType, str] | None = None,
    retries: int = 1,
) -> TargetCapabilities:
    """
    Probe both capabilities and modalities and return a combined result.

    Calls :func:`query_target_capabilities_async` and
    :func:`verify_target_modalities_async` and returns a
    :class:`TargetCapabilities` populated from the verified results, so
    callers don't need to assemble the dataclass themselves.

    Boolean capability flags not covered by
    :data:`_CAPABILITY_PROBES` (e.g. ``supports_editable_history``) are
    copied from ``target.capabilities`` (the target's declared native flags).

    Args:
        target (PromptTarget): The target to probe.
        per_probe_timeout_s (float): Per-attempt timeout (seconds) applied to
            each probe request.
        test_assets (dict[PromptDataType, str] | None): Mapping from non-text
            modality to a file path. See :func:`verify_target_modalities_async`.
        retries (int): Number of additional attempts after the first failure
            for each probe. Only exceptions/timeouts are retried; an explicit
            error response is final. Set to ``0`` to disable retries.
            Defaults to 1.

    Returns:
        TargetCapabilities: A dataclass reflecting verified capabilities and
        modalities. ``output_modalities`` is copied from
        ``target.capabilities.output_modalities`` because outputs cannot be
        verified by sending a request.
    """
    verified_caps = await query_target_capabilities_async(
        target=target, per_probe_timeout_s=per_probe_timeout_s, retries=retries
    )
    verified_modalities = await verify_target_modalities_async(
        target=target, test_assets=test_assets, per_probe_timeout_s=per_probe_timeout_s, retries=retries
    )

    declared = target.capabilities
    return TargetCapabilities(
        supports_multi_turn=CapabilityName.MULTI_TURN in verified_caps,
        supports_multi_message_pieces=CapabilityName.MULTI_MESSAGE_PIECES in verified_caps,
        supports_json_schema=CapabilityName.JSON_SCHEMA in verified_caps,
        supports_json_output=CapabilityName.JSON_OUTPUT in verified_caps,
        supports_editable_history=declared.supports_editable_history,
        supports_system_prompt=CapabilityName.SYSTEM_PROMPT in verified_caps,
        input_modalities=frozenset(verified_modalities),
        output_modalities=declared.output_modalities,
    )


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
