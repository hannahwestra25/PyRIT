# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.message_normalizer import GenericSystemSquashNormalizer, HistorySquashNormalizer
from pyrit.models import Message, MessagePiece
from pyrit.models.literals import ChatMessageRole
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration


_ADAPT_ALL = CapabilityHandlingPolicy(
    behaviors={
        CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
        CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.ADAPT,
        CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
        CapabilityName.JSON_OUTPUT: UnsupportedCapabilityBehavior.RAISE,
        CapabilityName.MULTI_MESSAGE_PIECES: UnsupportedCapabilityBehavior.RAISE,
        CapabilityName.EDITABLE_HISTORY: UnsupportedCapabilityBehavior.RAISE,
    }
)


def _make_message(role: ChatMessageRole, content: str) -> Message:
    return Message(message_pieces=[MessagePiece(role=role, original_value=content)])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_with_defaults_uses_raise_policy():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps)
    # Default policy is RAISE for all adaptable capabilities
    assert config.policy.get_behavior(capability=CapabilityName.MULTI_TURN) == UnsupportedCapabilityBehavior.RAISE


def test_init_with_explicit_policy():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    assert config.policy is _ADAPT_ALL


def test_init_all_supported_empty_pipeline():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    assert config.pipeline.normalizers == ()


def test_init_missing_capability_adapt_builds_pipeline():
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    assert len(config.pipeline.normalizers) == 2
    assert isinstance(config.pipeline.normalizers[0], GenericSystemSquashNormalizer)
    assert isinstance(config.pipeline.normalizers[1], HistorySquashNormalizer)


def test_init_missing_capability_raise_policy_raises():
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=True)
    with pytest.raises(ValueError, match="RAISE"):
        TargetConfiguration(capabilities=caps)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_capabilities_property():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps)
    assert config.capabilities is caps


# ---------------------------------------------------------------------------
# supports
# ---------------------------------------------------------------------------


def test_supports_returns_true_when_supported():
    caps = TargetCapabilities(supports_multi_turn=True)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    assert config.supports(capability=CapabilityName.MULTI_TURN) is True


def test_supports_returns_false_when_unsupported():
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    assert config.supports(capability=CapabilityName.MULTI_TURN) is False


# ---------------------------------------------------------------------------
# requires
# ---------------------------------------------------------------------------


def test_requires_passes_when_supported():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps)
    # Should not raise
    config.requires(capability=CapabilityName.MULTI_TURN)


def test_requires_passes_when_adapt():
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    # ADAPT policy → should not raise
    config.requires(capability=CapabilityName.MULTI_TURN)


def test_requires_raises_when_raise_policy():
    # Build with ADAPT so construction succeeds, then test requires() on a RAISE capability.
    # JSON_SCHEMA is RAISE and unsupported — but it's not normalizable, so construction
    # doesn't try to build a normalizer for it. Use a custom policy where system_prompt
    # is ADAPT (so pipeline builds), but then call requires() on JSON_OUTPUT which is RAISE.
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=False)
    policy = CapabilityHandlingPolicy(
        behaviors={
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.JSON_OUTPUT: UnsupportedCapabilityBehavior.RAISE,
        }
    )
    config = TargetConfiguration(capabilities=caps, policy=policy)
    # system_prompt is missing + ADAPT → requires passes
    config.requires(capability=CapabilityName.SYSTEM_PROMPT)
    # json_output is missing + RAISE → requires raises
    with pytest.raises(ValueError, match="RAISE"):
        config.requires(capability=CapabilityName.JSON_OUTPUT)


# ---------------------------------------------------------------------------
# normalize_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_async_passthrough_when_all_supported():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)
    msgs = [_make_message("user", "hello")]
    result = await config.normalize_async(messages=msgs)
    assert len(result) == 1
    assert result[0].message_pieces[0].converted_value == "hello"


@pytest.mark.asyncio
async def test_normalize_async_adapts_system_prompt():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)

    msgs = [
        _make_message("system", "you are helpful"),
        _make_message("user", "hello"),
    ]
    result = await config.normalize_async(messages=msgs)
    # System squash merges system into user messages — no system role left
    for msg in result:
        for piece in msg.message_pieces:
            assert piece.api_role != "system"


@pytest.mark.asyncio
async def test_normalize_async_adapts_multi_turn():
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps, policy=_ADAPT_ALL)

    msgs = [
        _make_message("user", "turn 1"),
        _make_message("assistant", "reply 1"),
        _make_message("user", "turn 2"),
    ]
    result = await config.normalize_async(messages=msgs)
    # History squash collapses into a single message
    assert len(result) == 1
    assert "[Conversation History]" in result[0].message_pieces[0].converted_value
    assert "turn 2" in result[0].message_pieces[0].converted_value
