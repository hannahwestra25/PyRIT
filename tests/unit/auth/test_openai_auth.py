# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import patch

import pytest

from pyrit.auth.openai_auth import resolve_openai_auth

AZURE_ENDPOINT = "https://my-resource.openai.azure.com/openai/v1"
NON_AZURE_ENDPOINT = "https://api.openai.com/v1"
API_KEY_ENV_VAR = "OPENAI_CHAT_API_KEY"


@pytest.fixture
def minted_provider():
    async def _provider() -> str:
        return "entra-token"

    with patch("pyrit.auth.openai_auth.get_azure_openai_auth", return_value=_provider) as mock_auth:
        yield _provider, mock_auth


def test_identity_ignores_env_var_api_key(minted_provider):
    """An explicit identity choice must not be downgraded to a key sitting in the environment."""
    provider, mock_auth = minted_provider
    with patch.dict(os.environ, {API_KEY_ENV_VAR: "sk-SECRET-FROM-DOTENV"}):
        resolved = resolve_openai_auth(
            endpoint=AZURE_ENDPOINT,
            api_key=None,
            api_key_environment_variable=API_KEY_ENV_VAR,
            auth_mode="identity",
        )

    assert resolved is provider
    mock_auth.assert_called_once_with(AZURE_ENDPOINT)


def test_identity_ignores_explicit_api_key(minted_provider):
    """Identity wins over a key passed alongside it rather than silently using the key."""
    provider, _ = minted_provider
    resolved = resolve_openai_auth(
        endpoint=AZURE_ENDPOINT,
        api_key="sk-explicit",
        api_key_environment_variable=API_KEY_ENV_VAR,
        auth_mode="identity",
    )

    assert resolved is provider


def test_identity_raises_for_non_azure_endpoint():
    with pytest.raises(ValueError, match="Identity-based authentication requires a recognized Azure"):
        resolve_openai_auth(
            endpoint=NON_AZURE_ENDPOINT,
            api_key=None,
            api_key_environment_variable=API_KEY_ENV_VAR,
            auth_mode="identity",
        )


def test_default_auth_mode_uses_env_var():
    with patch.dict(os.environ, {API_KEY_ENV_VAR: "sk-from-env"}):
        resolved = resolve_openai_auth(
            endpoint=AZURE_ENDPOINT,
            api_key=None,
            api_key_environment_variable=API_KEY_ENV_VAR,
        )

    assert resolved == "sk-from-env"


def test_api_key_mode_prefers_explicit_key_over_env_var():
    with patch.dict(os.environ, {API_KEY_ENV_VAR: "sk-from-env"}):
        resolved = resolve_openai_auth(
            endpoint=AZURE_ENDPOINT,
            api_key="sk-explicit",
            api_key_environment_variable=API_KEY_ENV_VAR,
            auth_mode="api_key",
        )

    assert resolved == "sk-explicit"


def test_api_key_mode_wraps_callable_before_reading_env_var():
    def sync_provider() -> str:
        return "callable-token"

    with patch.dict(os.environ, {API_KEY_ENV_VAR: "sk-from-env"}):
        resolved = resolve_openai_auth(
            endpoint=AZURE_ENDPOINT,
            api_key=sync_provider,
            api_key_environment_variable=API_KEY_ENV_VAR,
        )

    assert callable(resolved)
    assert resolved is not sync_provider


def test_api_key_mode_raises_when_no_key_available(minted_provider):
    """api_key mode no longer mints an Entra token just because the endpoint looks like Azure."""
    _, mock_auth = minted_provider
    with patch.dict(os.environ, {API_KEY_ENV_VAR: ""}):
        with pytest.raises(ValueError, match="No API key available"):
            resolve_openai_auth(
                endpoint=AZURE_ENDPOINT,
                api_key=None,
                api_key_environment_variable=API_KEY_ENV_VAR,
            )

    mock_auth.assert_not_called()


def test_api_key_mode_raises_for_non_azure_endpoint_without_key():
    with patch.dict(os.environ, {API_KEY_ENV_VAR: ""}):
        with pytest.raises(ValueError, match="No API key available"):
            resolve_openai_auth(
                endpoint=NON_AZURE_ENDPOINT,
                api_key=None,
                api_key_environment_variable=API_KEY_ENV_VAR,
            )


def test_api_key_mode_error_names_the_identity_migration():
    """The break is only safe if the error tells the caller how to opt into identity."""
    with patch.dict(os.environ, {API_KEY_ENV_VAR: ""}):
        with pytest.raises(ValueError) as exc_info:
            resolve_openai_auth(
                endpoint=AZURE_ENDPOINT,
                api_key=None,
                api_key_environment_variable=API_KEY_ENV_VAR,
            )

    message = str(exc_info.value)
    assert 'auth_mode="identity"' in message
    assert API_KEY_ENV_VAR in message
