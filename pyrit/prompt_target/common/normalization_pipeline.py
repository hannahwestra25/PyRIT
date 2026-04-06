# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import ClassVar

from pyrit.message_normalizer import (
    GenericSystemSquashNormalizer,
    HistorySquashNormalizer,
    MessageListNormalizer,
)
from pyrit.models import Message
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    NORMALIZABLE_CAPABILITIES,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)

logger = logging.getLogger(__name__)


def _default_normalizer_factory() -> dict[CapabilityName, MessageListNormalizer[Message]]:
    """
    Build the default normalizer for every normalizable capability.

    Returns:
        dict[CapabilityName, MessageListNormalizer[Message]]: Mapping from
        capability to its default normalizer instance.
    """
    return {
        CapabilityName.SYSTEM_PROMPT: GenericSystemSquashNormalizer(),
        CapabilityName.MULTI_TURN: HistorySquashNormalizer(),
    }


class ConversationNormalizationPipeline:
    """
    Ordered sequence of message normalizers that adapt conversations when
    the target lacks certain capabilities.

    The pipeline is constructed via ``from_capabilities``, which resolves
    capabilities and policy into a concrete, ordered tuple of normalizers.
    ``normalize_async`` then simply executes that tuple in order.
    """

    PIPELINE_ORDER: ClassVar[list[CapabilityName]] = [
        CapabilityName.SYSTEM_PROMPT,
        CapabilityName.MULTI_TURN,
    ]

    def __init__(self, normalizers: tuple[MessageListNormalizer[Message], ...] = ()) -> None:
        """
        Initialize the normalization pipeline with an ordered sequence of normalizers.

        Args:
            normalizers (tuple[MessageListNormalizer[Message], ...]):
                Ordered normalizers to apply during ``normalize_async``.
                Defaults to an empty tuple (pass-through).
        """
        self._normalizers = normalizers

    @classmethod
    def from_capabilities(
        cls,
        *,
        capabilities: TargetCapabilities,
        policy: CapabilityHandlingPolicy,
        normalizer_overrides: dict[CapabilityName, MessageListNormalizer[Message]] | None = None,
    ) -> "ConversationNormalizationPipeline":
        """
        Resolve capabilities and policy into a concrete pipeline of normalizers.

        For each capability in ``PIPELINE_ORDER``:

        * If the target already supports the capability, no normalizer is added.
        * If the capability is missing and the policy is ``ADAPT``, the
          corresponding normalizer (from overrides or defaults) is added.
        * If the capability is missing and the policy is ``RAISE``, a
          ``ValueError`` is raised immediately.

        Args:
            capabilities (TargetCapabilities): The target's declared capabilities.
            policy (CapabilityHandlingPolicy): How to handle each missing capability.
            normalizer_overrides (dict[CapabilityName, MessageListNormalizer[Message]] | None):
                Optional overrides for specific capability normalizers.
                Falls back to the defaults from ``_default_normalizer_factory``.

        Returns:
            ConversationNormalizationPipeline: A pipeline with the resolved
            ordered tuple of normalizers.

        Raises:
            ValueError: If a required capability is missing and the policy is RAISE,
                or if a capability is not normalizable, or if no normalizer is
                available for an ADAPT policy.
        """
        defaults = _default_normalizer_factory()
        overrides = normalizer_overrides or {}
        normalizers: list[MessageListNormalizer[Message]] = []

        for capability in cls.PIPELINE_ORDER:
            if capabilities.supports(capability=capability):
                continue

            if capability not in NORMALIZABLE_CAPABILITIES:
                raise ValueError(
                    f"Target does not support '{capability.value}' and this capability cannot be adapted."
                )

            behavior = policy.get_behavior(capability=capability)

            if behavior == UnsupportedCapabilityBehavior.RAISE:
                raise ValueError(
                    f"Target does not support '{capability.value}' and the handling policy is RAISE."
                )

            normalizer = overrides.get(capability) or defaults.get(capability)
            if normalizer is None:
                raise ValueError(
                    f"Target does not support '{capability.value}' and the policy is ADAPT, "
                    f"but no normalizer is available for this capability."
                )

            normalizers.append(normalizer)

        return cls(normalizers=tuple(normalizers))

    async def normalize_async(self, *, messages: list[Message]) -> list[Message]:
        """
        Run the pre-resolved normalizer sequence over the messages.

        Args:
            messages (list[Message]): The full conversation to normalize.

        Returns:
            list[Message]: The (possibly adapted) message list.
        """
        result = list(messages)
        for normalizer in self._normalizers:
            result = await normalizer.normalize_async(result)
        return result

    @property
    def normalizers(self) -> tuple[MessageListNormalizer[Message], ...]:
        """
        The ordered normalizers in this pipeline.

        Returns:
            tuple[MessageListNormalizer[Message], ...]: The normalizer sequence.
        """
        return self._normalizers
