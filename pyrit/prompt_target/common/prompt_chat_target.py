# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.common.deprecation import print_deprecation_message
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities


class PromptChatTarget(PromptTarget):
    """
    Deprecated. Use :class:`PromptTarget` directly instead.

    This class will be removed in 0.12.0.
    """

    _DEFAULT_CAPABILITIES: TargetCapabilities = TargetCapabilities(
        supports_multi_turn=True, supports_multi_message_pieces=True
    )

    def __init__(
        self,
        *,
        max_requests_per_minute: Optional[int] = None,
        endpoint: str = "",
        model_name: str = "",
        underlying_model: Optional[str] = None,
        custom_capabilities: Optional[TargetCapabilities] = None,
    ) -> None:
        print_deprecation_message(
            old_item=PromptChatTarget,
            new_item=PromptTarget,
            removed_in="0.12.0",
        )
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            endpoint=endpoint,
            model_name=model_name,
            underlying_model=underlying_model,
            custom_capabilities=custom_capabilities,
        )
