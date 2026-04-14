# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import warnings
from collections.abc import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion

from pyrit.identifiers import ComponentIdentifier
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.message_normalizer import GenericSystemSquashNormalizer
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import AzureMLChatTarget, OpenAIChatTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.openai.openai_response_target import OpenAIResponseTarget


def _make_message_piece(*, role: str, content: str, conversation_id: str = "conv1") -> MessagePiece:
    return MessagePiece(
        role=role,
        conversation_id=conversation_id,
        original_value=content,
        converted_value=content,
        original_value_data_type="text",
        converted_value_data_type="text",
        prompt_target_identifier=ComponentIdentifier(class_name="test", class_module="test"),
        attack_identifier=ComponentIdentifier(class_name="test", class_module="test"),
    )


def _make_message(*, role: str, content: str, conversation_id: str = "conv1") -> Message:
    return Message(message_pieces=[_make_message_piece(role=role, content=content, conversation_id=conversation_id)])


def _create_mock_chat_completion(content: str = "hi") -> MagicMock:
    mock = MagicMock(spec=ChatCompletion)
    mock.choices = [MagicMock()]
    mock.choices[0].finish_reason = "stop"
    mock.choices[0].message.content = content
    mock.choices[0].message.audio = None
    mock.choices[0].message.tool_calls = None
    mock.model_dump_json.return_value = json.dumps(
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    )
    return mock


# ---------------------------------------------------------------------------
# OpenAIChatTarget — normalize_async is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_openai_chat_target_calls_normalize_async():
    target = OpenAIChatTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
    )

    user_msg = _make_message(role="user", content="hello")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    target._memory = mock_memory

    mock_completion = _create_mock_chat_completion("world")
    target._async_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    with patch.object(target.configuration, "normalize_async", new_callable=AsyncMock) as mock_normalize:
        mock_normalize.return_value = [user_msg]
        await target.send_prompt_async(message=user_msg)

        mock_normalize.assert_called_once()
        call_messages = mock_normalize.call_args.kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0].get_value() == "hello"


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_openai_chat_target_sends_normalized_to_construct_request():
    """Verify that the normalized (not original) conversation is used for the API body."""
    target = OpenAIChatTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
    )

    user_msg = _make_message(role="user", content="original")
    adapted_msg = _make_message(role="user", content="adapted")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    target._memory = mock_memory

    mock_completion = _create_mock_chat_completion("response")
    target._async_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    with (
        patch.object(target.configuration, "normalize_async", new_callable=AsyncMock, return_value=[adapted_msg]),
        patch.object(
            target, "_construct_request_body", new_callable=AsyncMock, return_value={"model": "gpt-4o", "messages": []}
        ) as mock_construct,
    ):
        await target.send_prompt_async(message=user_msg)

        # _construct_request_body should receive the adapted message, not the original
        call_conv = mock_construct.call_args.kwargs["conversation"]
        assert len(call_conv) == 1
        assert call_conv[0].get_value() == "adapted"


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_openai_chat_target_memory_not_mutated():
    """Memory-backed conversation must not be altered by normalize_async."""
    target = OpenAIChatTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        custom_configuration=TargetConfiguration(
            capabilities=TargetCapabilities(
                supports_multi_turn=True,
                supports_system_prompt=False,
                supports_multi_message_pieces=True,
                input_modalities=frozenset({frozenset(["text"])}),
            ),
            policy=CapabilityHandlingPolicy(
                behaviors={
                    CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                    CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
                }
            ),
        ),
    )

    system_msg = _make_message(role="system", content="be nice")
    user_msg = _make_message(role="user", content="hello")

    # Memory returns a conversation with a system message
    memory_conversation: MutableSequence[Message] = [system_msg]

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = memory_conversation
    target._memory = mock_memory

    mock_completion = _create_mock_chat_completion("response")
    target._async_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    await target.send_prompt_async(message=user_msg)

    # Memory-backed conversation should still contain the system message
    assert len(memory_conversation) == 2  # system + appended user
    assert memory_conversation[0].get_piece().api_role == "system"


# ---------------------------------------------------------------------------
# OpenAIResponseTarget — normalize_async is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_openai_response_target_calls_normalize_async():
    target = OpenAIResponseTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
    )

    user_msg = _make_message(role="user", content="hello")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    target._memory = mock_memory

    # Mock the API to return a simple response (no tool calls)
    mock_response = MagicMock()
    mock_response.error = None
    mock_response.status = "completed"
    mock_response.output = [MagicMock()]
    mock_response.output[0].type = "message"
    mock_response.output[0].content = [MagicMock()]
    mock_response.output[0].content[0].type = "output_text"
    mock_response.output[0].content[0].text = "world"
    mock_response.model_dump_json.return_value = json.dumps(
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "world"}]}]}
    )
    target._async_client.responses.create = AsyncMock(return_value=mock_response)

    with patch.object(target.configuration, "normalize_async", new_callable=AsyncMock) as mock_normalize:
        mock_normalize.return_value = [user_msg]
        await target.send_prompt_async(message=user_msg)

        mock_normalize.assert_called_once()
        call_messages = mock_normalize.call_args.kwargs["messages"]
        assert len(call_messages) == 1


# ---------------------------------------------------------------------------
# AzureMLChatTarget — normalize_async is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_azure_ml_target_calls_normalize_async():
    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
    )

    user_msg = _make_message(role="user", content="hello")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    target._memory = mock_memory

    with (
        patch.object(target.configuration, "normalize_async", new_callable=AsyncMock) as mock_normalize,
        patch.object(target, "_complete_chat_async", new_callable=AsyncMock, return_value="response"),
    ):
        mock_normalize.return_value = [user_msg]
        await target.send_prompt_async(message=user_msg)

        mock_normalize.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_azure_ml_target_sends_normalized_to_complete_chat():
    """Normalized (not original) messages should be passed to _complete_chat_async."""
    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
    )

    user_msg = _make_message(role="user", content="original")
    adapted_msg = _make_message(role="user", content="adapted")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    target._memory = mock_memory

    with (
        patch.object(target.configuration, "normalize_async", new_callable=AsyncMock, return_value=[adapted_msg]),
        patch.object(target, "_complete_chat_async", new_callable=AsyncMock, return_value="response") as mock_chat,
    ):
        await target.send_prompt_async(message=user_msg)

        call_messages = mock_chat.call_args.kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0].get_value() == "adapted"


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_azure_ml_target_memory_not_mutated():
    """Memory should retain original messages after normalization."""
    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
        custom_configuration=TargetConfiguration(
            capabilities=TargetCapabilities(
                supports_multi_turn=True,
                supports_system_prompt=False,
                supports_multi_message_pieces=True,
                input_modalities=frozenset({frozenset(["text"])}),
            ),
            policy=CapabilityHandlingPolicy(
                behaviors={
                    CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                    CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
                }
            ),
        ),
    )

    system_msg = _make_message(role="system", content="be nice")
    user_msg = _make_message(role="user", content="hello")

    memory_conversation: MutableSequence[Message] = [system_msg]

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = memory_conversation
    target._memory = mock_memory

    with patch.object(target, "_complete_chat_async", new_callable=AsyncMock, return_value="response"):
        await target.send_prompt_async(message=user_msg)

    # Memory must still have original system message
    assert len(memory_conversation) == 2
    assert memory_conversation[0].get_piece().api_role == "system"


# ---------------------------------------------------------------------------
# AzureMLChatTarget — message_normalizer deprecation
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
def test_azure_ml_generic_system_squash_normalizer_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        target = AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            message_normalizer=GenericSystemSquashNormalizer(),
        )
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "message_normalizer is deprecated" in str(deprecation_warnings[0].message)


@pytest.mark.usefixtures("patch_central_database")
def test_azure_ml_generic_system_squash_normalizer_creates_adapt_configuration():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        target = AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            message_normalizer=GenericSystemSquashNormalizer(),
        )
    # The configuration should now have supports_system_prompt=False with ADAPT policy
    assert not target.capabilities.supports_system_prompt
    # Pipeline should have a system squash normalizer
    assert target.configuration.includes(capability=CapabilityName.MULTI_TURN)
    assert not target.configuration.includes(capability=CapabilityName.SYSTEM_PROMPT)


@pytest.mark.usefixtures("patch_central_database")
def test_azure_ml_generic_system_squash_normalizer_does_not_override_explicit_config():
    """If custom_configuration is already provided, message_normalizer deprecation should not override it."""
    custom_config = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_multi_message_pieces=True,
        )
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        target = AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            message_normalizer=GenericSystemSquashNormalizer(),
            custom_configuration=custom_config,
        )
    # Explicit custom_configuration should win
    assert target.capabilities.supports_system_prompt


@pytest.mark.asyncio
@pytest.mark.usefixtures("patch_central_database")
async def test_azure_ml_system_squash_via_configuration_pipeline():
    """End-to-end: GenericSystemSquashNormalizer-equivalent behavior via TargetConfiguration pipeline."""
    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
        custom_configuration=TargetConfiguration(
            capabilities=TargetCapabilities(
                supports_multi_turn=True,
                supports_system_prompt=False,
                supports_multi_message_pieces=True,
                input_modalities=frozenset({frozenset(["text"])}),
            ),
            policy=CapabilityHandlingPolicy(
                behaviors={
                    CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                    CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
                }
            ),
        ),
    )

    system_msg = _make_message(role="system", content="be nice")
    user_msg = _make_message(role="user", content="hello")

    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = [system_msg]
    target._memory = mock_memory

    with patch.object(target, "_complete_chat_async", new_callable=AsyncMock, return_value="response") as mock_chat:
        await target.send_prompt_async(message=user_msg)

        # _complete_chat_async should receive normalized messages (system squashed into user)
        call_messages = mock_chat.call_args.kwargs["messages"]
        roles = [m.get_piece().api_role for m in call_messages]
        assert "system" not in roles
        # The squashed message should contain the system content
        assert "be nice" in call_messages[0].get_value()
        assert "hello" in call_messages[0].get_value()
