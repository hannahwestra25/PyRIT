# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Canonical credential-selection mode shared by auth resolvers and targets.

Lives in ``pyrit.common`` rather than ``pyrit.prompt_target`` because
``pyrit.auth`` resolvers consume it and must not depend on the target layer.

Not to be confused with ``pyrit.cli._auth.AuthMode``, which names the *Azure
credential flow* the CLI should use (``"auto"`` / ``"azure_cli"`` /
``"device_code"`` / ``"none"``).
"""

from typing import Literal

__all__ = ["AUTH_MODES", "AuthMode"]

#: How a component chooses its credential.
#:
#: ``api_key`` resolves a key from the explicit argument or the component's API key
#: environment variable, falling back to an ambient Azure identity only when neither
#: is available. ``identity`` is an explicit caller choice: the key and its
#: environment variable are skipped entirely and the component authenticates with an
#: ambient Azure identity (e.g. a Microsoft Entra ID token minted for its own
#: endpoint).
AuthMode = Literal["api_key", "identity"]

AUTH_MODES: tuple[AuthMode, ...] = ("api_key", "identity")
