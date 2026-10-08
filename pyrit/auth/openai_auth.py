# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections.abc import Awaitable, Callable
from typing import cast

from pyrit.auth.azure_auth import ensure_async_token_provider, get_azure_openai_auth, is_azure_openai_endpoint
from pyrit.common import default_values
from pyrit.common.auth_mode import AuthMode


def resolve_openai_auth(
    *,
    endpoint: str,
    api_key: str | Callable[[], str | Awaitable[str]] | None,
    api_key_environment_variable: str,
    auth_mode: AuthMode = "api_key",
) -> str | Callable[[], Awaitable[str]]:
    """
    Resolve OpenAI authentication from an explicit identity choice, a key, or an environment variable.

    Args:
        endpoint (str): The OpenAI-compatible endpoint URL.
        api_key (str | Callable[[], str | Awaitable[str]] | None): The explicit API key or token provider.
        api_key_environment_variable (str): Environment variable to use when ``api_key`` is not provided.
        auth_mode (AuthMode): ``"identity"`` authenticates with a Microsoft Entra ID token and ignores
            ``api_key`` and its environment variable entirely. ``"api_key"`` (the default) resolves a
            token-provider callable, then an explicit key, then the environment variable.

    Returns:
        str | Callable[[], Awaitable[str]]: API key string or async-compatible token provider.

    Raises:
        ValueError: If identity auth is requested for an endpoint that is not a recognized Azure
            OpenAI endpoint, or if ``"api_key"`` auth is requested and no key is available.
    """
    # Identity is an explicit caller choice, so it must never be silently downgraded to a key
    # that merely happens to be present in the environment.
    if auth_mode == "identity":
        if not is_azure_openai_endpoint(endpoint):
            raise ValueError(
                f"Identity-based authentication requires a recognized Azure OpenAI / AI Foundry endpoint, "
                f"but got '{endpoint}'. Use api_key authentication for this endpoint, or pass your own "
                "token provider callable as api_key."
            )
        return get_azure_openai_auth(endpoint)

    if api_key is not None and callable(api_key):
        return cast("str | Callable[[], Awaitable[str]]", ensure_async_token_provider(api_key))

    api_key_value = default_values.get_non_required_value(
        env_var_name=api_key_environment_variable, passed_value=api_key
    )
    if api_key_value:
        return api_key_value

    raise ValueError(
        f"No API key available for endpoint '{endpoint}'. Set the {api_key_environment_variable} environment "
        'variable, pass api_key explicitly, or pass auth_mode="identity" to authenticate with Microsoft '
        "Entra ID on a recognized Azure OpenAI / AI Foundry endpoint."
    )
