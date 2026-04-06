# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from dataclasses import dataclass

from pyrit.message_normalizer import (
    GenericSystemSquashNormalizer,
    HistorySquashNormalizer,
    MessageListNormalizer,
)
from pyrit.models import Message
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NormalizerRegistryEntry:
    """Single entry in the normalizer registry."""

    order: int
    normalizer_factory: type[MessageListNormalizer[Message]]


# ---------------------------------------------------------------------------
# Single registry: add new normalizable capabilities here and nowhere else.
# ---------------------------------------------------------------------------
_NORMALIZER_REGISTRY: dict[CapabilityName, _NormalizerRegistryEntry] = {
    CapabilityName.SYSTEM_PROMPT: _NormalizerRegistryEntry(order=0, normalizer_factory=GenericSystemSquashNormalizer),
    CapabilityName.MULTI_TURN: _NormalizerRegistryEntry(order=1, normalizer_factory=HistorySquashNormalizer),
}

# Derived constants — no manual maintenance required.
NORMALIZABLE_CAPABILITIES: frozenset[CapabilityName] = frozenset(_NORMALIZER_REGISTRY)

_PIPELINE_ORDER: list[CapabilityName] = sorted(
    _NORMALIZER_REGISTRY,
    key=lambda cap: _NORMALIZER_REGISTRY[cap].order,
)


def _default_normalizers() -> dict[CapabilityName, MessageListNormalizer[Message]]:
    """
    Build a fresh default normalizer instance for every registered capability.

    Returns:
        dict[CapabilityName, MessageListNormalizer[Message]]: Mapping from
        capability to a new default normalizer instance.
    """
    return {cap: entry.normalizer_factory() for cap, entry in _NORMALIZER_REGISTRY.items()}


class ConversationNormalizationPipeline:
    """
    Ordered sequence of message normalizers that adapt conversations when
    the target lacks certain capabilities.

    The pipeline is constructed via ``from_capabilities``, which resolves
    capabilities and policy into a concrete, ordered tuple of normalizers.
    ``normalize_async`` then simply executes that tuple in order.

    To add a new normalizable capability, add a single entry to
    ``_NORMALIZER_REGISTRY``.  ``NORMALIZABLE_CAPABILITIES``,
    pipeline ordering, and default normalizers are all derived from it.
    """

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

        NOTE: Normalizers are only valid when the capability can be overridden with a normalizer (which is indicated
        by its presence in the registry), so we only iterate over valid capabilities in this function and add normalizers
        only when the capability can support normalization.

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
        defaults = _default_normalizers()
        overrides = normalizer_overrides or {}
        normalizers: list[MessageListNormalizer[Message]] = []

        for capability in _PIPELINE_ORDER:
            if capabilities.supports(capability=capability):
                continue

            behavior = policy.get_behavior(capability=capability)

            if behavior == UnsupportedCapabilityBehavior.RAISE:
                raise ValueError(f"Target does not support '{capability.value}' and the handling policy is RAISE.")

            normalizer = overrides.get(capability)
            if normalizer is None:
                normalizer = defaults.get(capability)
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
