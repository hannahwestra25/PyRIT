# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public-safe classification for exception-derived API fields."""

from collections.abc import Sequence
from dataclasses import dataclass

from pyrit.models import RetryEvent

_RATE_LIMIT_TYPES = frozenset(
    {
        "RateLimitError",
        "RateLimitException",
        "TooManyRequestsError",
    }
)
_TIMEOUT_TYPES = frozenset(
    {
        "APITimeoutError",
        "ServiceRequestTimeoutError",
        "ServiceResponseTimeoutError",
        "Timeout",
        "TimeoutError",
    }
)
_AUTHENTICATION_TYPES = frozenset(
    {
        "AuthenticationError",
        "AuthorizationError",
        "ClientAuthenticationError",
        "CredentialUnavailableError",
        "PermissionDeniedError",
    }
)
_UNAVAILABLE_TYPES = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "NetworkError",
        "ServerErrorException",
        "ServiceRequestError",
        "ServiceResponseError",
        "ServiceUnavailableError",
    }
)
_BLOCKED_TYPES = frozenset(
    {
        "ContentFilterFinishReasonError",
        "ScorerLLMResponseBlockedException",
    }
)

_COMPONENT_SUBJECTS = {
    "objective_target": "The objective target",
    "adversarial_chat": "The adversarial target",
    "objective_scorer": "The objective scorer",
    "objective_scorer_target": "The objective scorer target",
    "refusal_scorer": "The refusal scorer",
    "refusal_scorer_target": "The refusal scorer target",
    "auxiliary_scorer": "The auxiliary scorer",
    "auxiliary_scorer_target": "The auxiliary scorer target",
    "converter": "The converter",
    "converter_target": "The converter target",
}


@dataclass(frozen=True)
class PublicError:
    """Stable exception category and message safe for API clients."""

    error_type: str
    message: str


def classify_public_error(
    *,
    exception_type: str | None,
    retry_events: Sequence[RetryEvent] = (),
    default_error_type: str,
    default_message: str,
    default_subject: str,
) -> PublicError:
    """
    Classify an error using exception class names and retry context.

    Raw exception messages are intentionally excluded because they may contain
    credentials, endpoints, paths, or provider diagnostics.

    Args:
        exception_type: Exception class name captured for the failed operation.
        retry_events: Structured retries that may retain the root exception type.
        default_error_type: Stable type used when no category is recognized.
        default_message: Stable message used when no category is recognized.
        default_subject: Public-safe subject used when no component role is available.

    Returns:
        A public-safe error type and actionable message.
    """
    primary = _classify_exception_type(exception_type=exception_type, subject=default_subject)
    retry_classifications = [
        _classify_exception_type(
            exception_type=event.exception_type,
            subject=_COMPONENT_SUBJECTS.get(event.component_role, default_subject),
        )
        for event in reversed(retry_events)
    ]

    if primary:
        contextual_primary = next(
            (
                classified
                for classified in retry_classifications
                if classified and classified.error_type == primary.error_type
            ),
            None,
        )
        return contextual_primary or primary

    retry_classification = next((classified for classified in retry_classifications if classified), None)
    if retry_classification:
        return retry_classification

    return PublicError(error_type=default_error_type, message=default_message)


def sanitize_retry_event(event: RetryEvent) -> RetryEvent:
    """
    Create an API-safe retry event while retaining useful execution context.

    Args:
        event: Persisted retry event containing internal diagnostics.

    Returns:
        A copy with a curated category/message and no endpoint.
    """
    public_error = classify_public_error(
        exception_type=event.exception_type,
        default_error_type="RetryableOperationError",
        default_message="Retryable operation failed.",
        default_subject=_COMPONENT_SUBJECTS.get(event.component_role, "The operation"),
    )
    return event.model_copy(
        update={
            "exception_type": public_error.error_type,
            "exception_message": public_error.message,
            "endpoint": None,
        }
    )


def _classify_exception_type(*, exception_type: str | None, subject: str) -> PublicError | None:
    if exception_type in _RATE_LIMIT_TYPES:
        return PublicError(
            error_type="RateLimitedError",
            message=f"{subject} was rate limited. Please try again later.",
        )
    if exception_type in _TIMEOUT_TYPES:
        return PublicError(
            error_type="OperationTimeoutError",
            message=f"{subject} timed out. Please try again.",
        )
    if exception_type in _AUTHENTICATION_TYPES:
        return PublicError(
            error_type="AuthenticationFailedError",
            message=f"{subject} could not authenticate. Verify its credentials and permissions.",
        )
    if exception_type in _UNAVAILABLE_TYPES:
        return PublicError(
            error_type="ServiceUnavailableError",
            message=f"{subject} is unavailable. Verify connectivity and try again.",
        )
    if exception_type in _BLOCKED_TYPES:
        return PublicError(
            error_type="ContentBlockedError",
            message=f"{subject} response was blocked by content policy.",
        )
    return None
