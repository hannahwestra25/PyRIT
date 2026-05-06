# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any

from pyrit.prompt_target.common.prompt_target import PromptTarget


class PromptChatTarget(PromptTarget):
    """
    .. deprecated:: 0.14.0
        ``PromptChatTarget`` is deprecated and will be removed in v0.15.0. Use
        :class:`PromptTarget` directly with a ``TargetConfiguration`` declaring
        ``supports_multi_turn=True`` and ``supports_editable_history=True``.

    Backwards-compatible alias for :class:`PromptTarget`. All chat-target functionality
    (``set_system_prompt``, ``is_response_format_json``) lives on :class:`PromptTarget`.
    Subclassing or instantiating this class emits a :class:`DeprecationWarning`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"Subclassing PromptChatTarget is deprecated and will be removed in v0.15.0. "
            f"Inherit from PromptTarget directly and declare supports_multi_turn=True and "
            f"supports_editable_history=True in your _DEFAULT_CONFIGURATION. "
            f"({cls.__name__})",
            DeprecationWarning,
            stacklevel=2,
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "PromptChatTarget is deprecated and will be removed in v0.15.0. "
            "Use PromptTarget directly with a TargetConfiguration declaring "
            "supports_multi_turn=True and supports_editable_history=True.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
