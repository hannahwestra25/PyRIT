# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pyrit.executor.core import StrategyConverterConfig
from pyrit.models import JsonSchemaDefinition, SeedPrompt
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer, TrueFalseScorer

logger = logging.getLogger(__name__)

# Default first message sent to the adversarial chat when there is no objective-target
# response yet (rendered with ``{{ objective }}``).
DEFAULT_ADVERSARIAL_FIRST_MESSAGE = "Generate your first message to achieve: {{ objective }}"

# Default per-turn template handed to the adversarial chat. The manager computes the actual
# feedback text in Python (handling blocked/error/empty responses and optional score feedback)
# and exposes it to the template as ``feedback_text``; the default simply renders it. Custom
# templates may wrap ``feedback_text`` and reference ``objective``, and are rendered strictly, so
# a reference to any other variable raises rather than silently producing empty output.
DEFAULT_ADVERSARIAL_PROMPT_TEMPLATE = "{{ feedback_text }}"


def resolve_adversarial_json_schema(
    *,
    system_prompt: SeedPrompt | None,
    first_message: SeedPrompt | None,
) -> JsonSchemaDefinition | None:
    """
    Resolve the single adversarial-chat response JSON schema from a pair of prompts.

    The schema may be declared on either the adversarial system prompt or the first message
    (via ``response_json_schema`` / ``response_json_schema_name`` in YAML), but not both —
    declaring it twice is ambiguous about which one drives the response shape.

    Args:
        system_prompt: The resolved adversarial system-prompt SeedPrompt, or None.
        first_message: The resolved adversarial first-message SeedPrompt, or None.

    Returns:
        The declared schema, or None when neither prompt declares one.

    Raises:
        ValueError: If both prompts declare a ``response_json_schema``.
    """
    system_schema = system_prompt.response_json_schema if system_prompt is not None else None
    first_message_schema = first_message.response_json_schema if first_message is not None else None
    if system_schema is not None and first_message_schema is not None:
        raise ValueError(
            "Both the adversarial system prompt and first message declare a response_json_schema; "
            "set the schema on only one of them."
        )
    return system_schema or first_message_schema


@dataclass
class AttackAdversarialConfig:
    """
    Adversarial configuration for attacks that involve adversarial chat targets.

    This class defines the configuration for attacks that utilize an adversarial chat target,
    including the target chat model, system prompt, and seed prompt for the attack.
    """

    # Adversarial chat target for the attack
    target: PromptTarget

    # First message sent to the adversarial chat when there is no objective-target response
    # yet (supports the {{ objective }} template variable). May be None for strategies that
    # do not use a first message.
    first_message: str | SeedPrompt | None = DEFAULT_ADVERSARIAL_FIRST_MESSAGE

    # Template rendered each turn to wrap the per-turn feedback text the manager computes from
    # the objective target's latest response. Receives ``feedback_text`` and ``objective``.
    adversarial_prompt_template: str | SeedPrompt | None = DEFAULT_ADVERSARIAL_PROMPT_TEMPLATE

    # System prompt for the adversarial chat target, as an inline Jinja template string or a
    # SeedPrompt. When None, the attack's own built-in default system prompt is used instead.
    system_prompt: str | SeedPrompt | None = None

    # Optional additional instructions layered on top of the resolved system prompt above (either
    # ``system_prompt`` or, when that is None, the attack's built-in default). Lets callers add
    # extra rules (e.g. "never use copy-through attacks") on top of a technique's default persona
    # without hand-copying it into a full replacement string. Supports the same Jinja template
    # variables as ``system_prompt`` (e.g. ``{{ objective }}``).
    system_prompt_addendum: str | SeedPrompt | None = None


def _coerce_and_validate_prompt_component(
    *,
    value: str | SeedPrompt,
    required_parameters: list[str],
    error_message: str | None,
    component_name: str = "system prompt",
) -> SeedPrompt:
    """
    Coerce an inline string or ``SeedPrompt`` into a ``SeedPrompt`` declaring ``required_parameters``.

    Inline strings are trusted: they are wrapped in a Jinja ``SeedPrompt`` whose declared
    parameters are set to ``required_parameters``. Explicitly provided ``SeedPrompt`` objects are
    validated against ``required_parameters`` and returned unchanged (never copied).

    Args:
        value: The inline string or SeedPrompt to coerce.
        required_parameters: Parameter names the resolved template must support.
        error_message: Optional custom error message for validation failures.
        component_name: Human-readable label for the component being validated, used only in the
            default failure message (ignored when ``error_message`` is provided) so callers can
            tell which prompt component failed — e.g. "system prompt" vs. "system_prompt_addendum".

    Returns:
        The resolved SeedPrompt.

    Raises:
        ValueError: If an explicitly provided SeedPrompt is missing required parameters.
    """
    if isinstance(value, SeedPrompt):
        # Validate only explicitly provided SeedPrompts against the required parameters.
        declared = value.parameters or []
        missing = [param for param in required_parameters if param not in declared]
        if missing:
            raise ValueError(error_message or f"Adversarial {component_name} is missing required parameters: {missing}")
        return value

    # Inline strings are trusted — declare all required params so Jinja rendering works.
    return SeedPrompt(
        value=value,
        is_jinja_template=True,
        parameters=list(required_parameters),
    )


def resolve_adversarial_system_prompt(
    *,
    config: AttackAdversarialConfig,
    default_system_prompt_path: str | Path,
    required_parameters: list[str],
    error_message: str | None = None,
) -> SeedPrompt:
    """
    Resolve the effective adversarial system-prompt ``SeedPrompt`` for a strategy.

    Resolution order:

    1. ``config.system_prompt`` (inline string or SeedPrompt), if provided.
    2. ``default_system_prompt_path``.

    Inline strings are trusted: they are wrapped in a Jinja ``SeedPrompt`` whose declared
    parameters are set to ``required_parameters``. Explicitly provided ``SeedPrompt`` objects
    and YAML files are validated against ``required_parameters``.

    When ``config.system_prompt_addendum`` is also set, its resolved text is appended to
    whichever base prompt was resolved above (separated by a blank line), producing a single
    combined ``SeedPrompt`` that declares the same ``required_parameters``. The combined prompt's
    ``response_json_schema`` is whichever of the base prompt or the addendum declares one (it is
    an error for both to declare one — see Raises below). This lets callers layer extra
    instructions on top of a technique's default (or a custom ``system_prompt``) instead of
    hand-copying the base text into a full replacement string.

    Addendum validation failures never reuse ``error_message`` (that message is written for the
    base prompt's specific contract, e.g. "must have an objective") — the addendum always raises
    a generic message naming ``system_prompt_addendum`` so failures aren't misattributed to the
    base prompt.

    Args:
        config: The adversarial configuration to resolve the system prompt from.
        default_system_prompt_path: Fallback YAML path when neither inline nor path is set.
        required_parameters: Parameter names the resolved template must support.
        error_message: Optional custom error message for base-prompt validation failures.

    Returns:
        The resolved adversarial system-prompt SeedPrompt.

    Raises:
        ValueError: If an explicitly provided SeedPrompt (base or addendum) is missing required
            parameters, or if both the base prompt and the addendum declare a
            ``response_json_schema``.
    """
    system_prompt = config.system_prompt
    if system_prompt is not None:
        base_prompt = _coerce_and_validate_prompt_component(
            value=system_prompt,
            required_parameters=required_parameters,
            error_message=error_message,
        )
    else:
        base_prompt = SeedPrompt.from_yaml_with_required_parameters(
            template_path=default_system_prompt_path,
            required_parameters=required_parameters,
            error_message=error_message,
        )

    if config.system_prompt_addendum is None:
        return base_prompt

    addendum_prompt = _coerce_and_validate_prompt_component(
        value=config.system_prompt_addendum,
        required_parameters=required_parameters,
        error_message=None,
        component_name="system_prompt_addendum",
    )
    if base_prompt.response_json_schema is not None and addendum_prompt.response_json_schema is not None:
        raise ValueError(
            "Both the resolved adversarial system prompt and system_prompt_addendum declare a "
            "response_json_schema; set the schema on only one of them."
        )
    return SeedPrompt(
        value=f"{base_prompt.value}\n\n{addendum_prompt.value}",
        is_jinja_template=True,
        parameters=list(required_parameters),
        response_json_schema=base_prompt.response_json_schema or addendum_prompt.response_json_schema,
    )


@dataclass
class AttackScoringConfig:
    """
    Scoring configuration for evaluating attack effectiveness.

    This class defines the scoring components used to evaluate attack effectiveness,
    detect refusals, and perform auxiliary scoring operations.
    """

    # Primary scorer for evaluating attack effectiveness
    objective_scorer: TrueFalseScorer | None = None

    # Refusal scorer for detecting refusals or non-compliance
    refusal_scorer: TrueFalseScorer | None = None

    # Additional scorers for auxiliary metrics or custom evaluations
    auxiliary_scorers: list[Scorer] = field(default_factory=list)

    # Whether to use scoring results as feedback for iterative attacks
    use_score_as_feedback: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If the objective or refusal scorers are not of type TrueFalseScorer.
        """
        # Enforce objective scorer type: must be a true/false scorer if provided
        if self.objective_scorer and not isinstance(self.objective_scorer, TrueFalseScorer):
            raise ValueError("Objective scorer must be a TrueFalseScorer")

        # Enforce refusal scorer type: must be a true/false scorer if provided
        if self.refusal_scorer and not isinstance(self.refusal_scorer, TrueFalseScorer):
            raise ValueError("Refusal scorer must be a TrueFalseScorer")


@dataclass
class AttackConverterConfig(StrategyConverterConfig):
    """
    Configuration for converters used in attacks.

    This class defines the converter configurations that transform prompts
    during the attack process, both for requests and responses.
    """
