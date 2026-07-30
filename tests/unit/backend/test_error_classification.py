# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for public-safe backend error classification."""

import pytest

from pyrit.backend.error_classification import classify_public_error, sanitize_retry_event
from pyrit.models import RetryEvent


@pytest.mark.parametrize(
    ("exception_type", "expected_type", "expected_message"),
    [
        ("RateLimitError", "RateLimitedError", "The objective target was rate limited. Please try again later."),
        ("APITimeoutError", "OperationTimeoutError", "The objective target timed out. Please try again."),
        (
            "AuthenticationError",
            "AuthenticationFailedError",
            "The objective target could not authenticate. Verify its credentials and permissions.",
        ),
        (
            "APIConnectionError",
            "ServiceUnavailableError",
            "The objective target is unavailable. Verify connectivity and try again.",
        ),
        (
            "ScorerLLMResponseBlockedException",
            "ContentBlockedError",
            "The objective target response was blocked by content policy.",
        ),
    ],
)
def test_sanitize_retry_event_classifies_known_exception(
    exception_type: str,
    expected_type: str,
    expected_message: str,
) -> None:
    event = RetryEvent(
        exception_type=exception_type,
        exception_message=r"provider secret=sk-test at C:\internal\provider.py",
        component_role="objective_target",
        component_name="OpenAIChatTarget",
        endpoint="https://provider.internal/?api_key=sk-test",
    )

    sanitized = sanitize_retry_event(event)

    assert sanitized.exception_type == expected_type
    assert sanitized.exception_message == expected_message
    assert sanitized.component_role == "objective_target"
    assert sanitized.component_name == "OpenAIChatTarget"
    assert sanitized.endpoint is None
    assert "sk-test" not in sanitized.model_dump_json()
    assert "provider.internal" not in sanitized.model_dump_json()


def test_classify_public_error_uses_retry_root_cause() -> None:
    event = RetryEvent(
        exception_type="RateLimitError",
        exception_message="provider deployment secret-model was throttled",
        component_role="objective_scorer",
    )

    classified = classify_public_error(
        exception_type="RetryError",
        retry_events=[event],
        default_error_type="AttackExecutionError",
        default_message="Attack execution failed.",
        default_subject="Attack execution",
    )

    assert classified.error_type == "RateLimitedError"
    assert classified.message == "The objective scorer was rate limited. Please try again later."
    assert "secret-model" not in classified.message


def test_classify_public_error_keeps_unknown_exception_generic() -> None:
    classified = classify_public_error(
        exception_type="ProviderFailure",
        retry_events=[
            RetryEvent(
                exception_type="ProviderRetryFailure",
                exception_message=r"secret=sk-test C:\internal\provider.py",
                component_role="objective_target",
            )
        ],
        default_error_type="AttackExecutionError",
        default_message="Attack execution failed.",
        default_subject="Attack execution",
    )

    assert classified.error_type == "AttackExecutionError"
    assert classified.message == "Attack execution failed."
