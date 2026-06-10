# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Notebook Upgrade Helper for PyRIT.

When users upgrade PyRIT versions, their existing notebooks may break due to
renamed classes, moved modules, or changed function signatures. This module
provides an IPython exception handler that detects common upgrade-related
errors and displays actionable fix suggestions inline in the notebook.

Usage::

    from pyrit.common.notebook_upgrade_helper import enable_upgrade_helper

    enable_upgrade_helper()   # Start showing upgrade suggestions on errors
    # ... run your notebook cells ...
    disable_upgrade_helper()  # Stop (optional)

The helper is purely additive: the original traceback is always displayed
first, and suggestions appear below it. If the helper itself encounters an
error, it silently falls back to default behavior.
"""

from __future__ import annotations

import logging
import re
import traceback as tb_module
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration rule definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationRule:
    """A single migration rule describing a breaking change between PyRIT versions.

    Attributes:
        error_types: Exception types this rule can match (e.g., ImportError).
        old_ref: Human-readable description of the old usage.
        new_ref: Human-readable description of the replacement.
        version_introduced: The PyRIT version that introduced this change.
        description: Explanation of what changed and why.
        example_old: Example code using the old API.
        example_new: Example code using the new API.
        match: Callable that inspects the exception and traceback lines
            to decide if this rule applies. Returns True if matched.
    """

    error_types: tuple[type[BaseException], ...]
    old_ref: str
    new_ref: str
    version_introduced: str
    description: str
    example_old: str
    example_new: str
    match: Any = field(repr=False)  # Callable[[BaseException, list[str]], bool]


def _is_pyrit_traceback(tb_lines: list[str]) -> bool:
    """Check whether the traceback involves PyRIT code."""
    return any("pyrit" in line.lower() for line in tb_lines)


def _make_import_rule(
    *,
    old_module: str,
    old_name: str,
    new_module: str,
    new_name: str | None = None,
    version: str,
    description: str,
) -> MigrationRule:
    """Helper to build a MigrationRule for a moved/renamed import."""
    actual_new_name = new_name or old_name

    pattern = re.compile(
        rf"cannot import name '{re.escape(old_name)}' from '{re.escape(old_module)}'"
    )

    def matcher(exc: BaseException, tb_lines: list[str]) -> bool:
        return bool(pattern.search(str(exc)))

    return MigrationRule(
        error_types=(ImportError,),
        old_ref=f"{old_module}.{old_name}",
        new_ref=f"{new_module}.{actual_new_name}",
        version_introduced=version,
        description=description,
        example_old=f"from {old_module} import {old_name}",
        example_new=f"from {new_module} import {actual_new_name}",
        match=matcher,
    )


def _make_module_not_found_rule(
    *,
    old_module: str,
    new_module: str,
    version: str,
    description: str,
) -> MigrationRule:
    """Helper to build a MigrationRule for a removed/renamed module."""
    pattern = re.compile(rf"No module named '{re.escape(old_module)}'")

    def matcher(exc: BaseException, tb_lines: list[str]) -> bool:
        return bool(pattern.search(str(exc)))

    return MigrationRule(
        error_types=(ModuleNotFoundError,),
        old_ref=old_module,
        new_ref=new_module,
        version_introduced=version,
        description=description,
        example_old=f"import {old_module}",
        example_new=f"import {new_module}",
        match=matcher,
    )


def _make_attribute_rule(
    *,
    module: str,
    old_name: str,
    new_module: str | None = None,
    new_name: str,
    version: str,
    description: str,
) -> MigrationRule:
    """Helper to build a MigrationRule for a renamed attribute/class."""
    actual_new_module = new_module or module
    pattern = re.compile(
        rf"module '{re.escape(module)}' has no attribute '{re.escape(old_name)}'"
    )

    def matcher(exc: BaseException, tb_lines: list[str]) -> bool:
        return bool(pattern.search(str(exc))) and _is_pyrit_traceback(tb_lines)

    return MigrationRule(
        error_types=(AttributeError,),
        old_ref=f"{module}.{old_name}",
        new_ref=f"{actual_new_module}.{new_name}",
        version_introduced=version,
        description=description,
        example_old=f"from {module} import {old_name}",
        example_new=f"from {actual_new_module} import {new_name}",
        match=matcher,
    )


def _make_kwarg_rule(
    *,
    callable_pattern: str,
    old_kwarg: str,
    new_kwarg: str,
    version: str,
    description: str,
) -> MigrationRule:
    """Helper to build a MigrationRule for a removed/renamed keyword argument."""
    pattern = re.compile(
        rf"{re.escape(callable_pattern)}.*got an unexpected keyword argument '{re.escape(old_kwarg)}'"
    )

    def matcher(exc: BaseException, tb_lines: list[str]) -> bool:
        if not pattern.search(str(exc)):
            return False
        return _is_pyrit_traceback(tb_lines)

    return MigrationRule(
        error_types=(TypeError,),
        old_ref=f"{callable_pattern}({old_kwarg}=...)",
        new_ref=f"{callable_pattern}({new_kwarg}=...)",
        version_introduced=version,
        description=description,
        example_old=f"{callable_pattern}({old_kwarg}=value)",
        example_new=f"{callable_pattern}({new_kwarg}=value)",
        match=matcher,
    )


# ---------------------------------------------------------------------------
# Migration rule registry
# ---------------------------------------------------------------------------
# Migration rule registry (auto-built from shared migration data)
# ---------------------------------------------------------------------------

from pyrit.common._migration_data import (
    IMPORT_RENAMES as _IMPORT_RENAMES_DATA,
    MODULE_RENAMES as _MODULE_RENAMES_DATA,
)

# Build MIGRATION_RULES automatically from shared data.
# Import rules and module-not-found rules are generated; kwarg rules
# are kept as-is since they need specific callable_pattern values that
# don't map 1:1 from the shared data.
MIGRATION_RULES: list[MigrationRule] = []

# Module-not-found rules (bare "import pyrit.orchestrator" etc.)
for _mr in _MODULE_RENAMES_DATA:
    MIGRATION_RULES.append(
        _make_module_not_found_rule(
            old_module=_mr.old_module,
            new_module=_mr.new_module,
            version=_mr.removed_in or "unknown",
            description=f"Module '{_mr.old_module}' was removed. Use '{_mr.new_module}' instead.",
        )
    )

# Import rules (from X import Y)
for _ir in _IMPORT_RENAMES_DATA:
    MIGRATION_RULES.append(
        _make_import_rule(
            old_module=_ir.old_module,
            old_name=_ir.old_name,
            new_module=_ir.new_module,
            new_name=_ir.new_name,
            version=_ir.removed_in or "unknown",
            description=f"'{_ir.old_module}.{_ir.old_name}' moved to '{_ir.new_module}.{_ir.new_name}'.",
        )
    )

# Kwarg rules (manually maintained — these are specific to callable patterns
# and cannot be auto-derived from deprecation shims)
MIGRATION_RULES.extend([
    _make_kwarg_rule(
        callable_pattern="RedTeamingAttack",
        old_kwarg="prompt_target",
        new_kwarg="objective_target",
        version="0.13.0",
        description="'prompt_target' was renamed to 'objective_target'.",
    ),
    _make_kwarg_rule(
        callable_pattern="CrescendoAttack",
        old_kwarg="prompt_target",
        new_kwarg="objective_target",
        version="0.13.0",
        description="'prompt_target' was renamed to 'objective_target'.",
    ),
    _make_kwarg_rule(
        callable_pattern="PAIRAttack",
        old_kwarg="prompt_target",
        new_kwarg="objective_target",
        version="0.13.0",
        description="'prompt_target' was renamed to 'objective_target'.",
    ),
    _make_kwarg_rule(
        callable_pattern="RedTeamingAttack",
        old_kwarg="red_teaming_chat",
        new_kwarg="adversarial_chat",
        version="0.13.0",
        description="'red_teaming_chat' was renamed to 'adversarial_chat'.",
    ),
    _make_kwarg_rule(
        callable_pattern="CrescendoAttack",
        old_kwarg="red_teaming_chat",
        new_kwarg="adversarial_chat",
        version="0.13.0",
        description="'red_teaming_chat' was renamed to 'adversarial_chat'.",
    ),
    _make_kwarg_rule(
        callable_pattern="CrescendoAttack",
        old_kwarg="max_rounds",
        new_kwarg="max_turns",
        version="0.13.0",
        description="'max_rounds' was renamed to 'max_turns'.",
    ),
    _make_kwarg_rule(
        callable_pattern="PAIRAttack",
        old_kwarg="max_conversation_depth",
        new_kwarg="max_turns",
        version="0.13.0",
        description="'max_conversation_depth' was renamed to 'max_turns'.",
    ),
    _make_kwarg_rule(
        callable_pattern="RedTeamingAttack",
        old_kwarg="attack_strategy",
        new_kwarg="objective",
        version="0.13.0",
        description="'attack_strategy' was renamed to 'objective'.",
    ),
    _make_kwarg_rule(
        callable_pattern="RedTeamingAttack",
        old_kwarg="scorer",
        new_kwarg="objective_scorer",
        version="0.13.0",
        description="'scorer' was renamed to 'objective_scorer'.",
    ),
    _make_kwarg_rule(
        callable_pattern="CrescendoAttack",
        old_kwarg="scorer",
        new_kwarg="objective_scorer",
        version="0.13.0",
        description="'scorer' was renamed to 'objective_scorer'.",
    ),
    _make_kwarg_rule(
        callable_pattern="PAIRAttack",
        old_kwarg="scorer",
        new_kwarg="objective_scorer",
        version="0.13.0",
        description="'scorer' was renamed to 'objective_scorer'.",
    ),
])


def add_migration_rule(rule: MigrationRule) -> None:
    """Register a custom migration rule.

    This allows downstream projects or plugins to add their own rules for
    custom modules that wrap PyRIT.

    Args:
        rule: The migration rule to add.
    """
    MIGRATION_RULES.append(rule)


# ---------------------------------------------------------------------------
# Error matching
# ---------------------------------------------------------------------------


@dataclass
class UpgradeSuggestion:
    """A matched upgrade suggestion ready for display."""

    rule: MigrationRule
    original_error: BaseException


def suggest_fix(
    exc: BaseException,
    tb_lines: list[str] | None = None,
) -> UpgradeSuggestion | None:
    """Match an exception against known migration rules.

    Args:
        exc: The exception that was raised.
        tb_lines: Formatted traceback lines. If None, only the exception
            message is used for matching.

    Returns:
        An UpgradeSuggestion if a matching rule is found, otherwise None.
    """
    if tb_lines is None:
        tb_lines = []

    for rule in MIGRATION_RULES:
        if not isinstance(exc, rule.error_types):
            continue
        try:
            if rule.match(exc, tb_lines):
                return UpgradeSuggestion(rule=rule, original_error=exc)
        except Exception:
            # Never let a broken rule matcher crash the helper
            logger.debug("Migration rule match failed", exc_info=True)
            continue

    return None


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

_PLAIN_TEMPLATE = """\
┌──────────────────────────────────────────────────────────────────┐
│  ⚠️  PyRIT Upgrade Suggestion                                    │
├──────────────────────────────────────────────────────────────────┤
│  {description}
│  Changed in: v{version}
│
│  Replace:
│    - {example_old}
│    + {example_new}
└──────────────────────────────────────────────────────────────────┘"""

_HTML_TEMPLATE = """\
<div style="border: 2px solid #e8a735; border-radius: 8px; padding: 12px 16px;
            margin: 8px 0; background: #fff8e1; font-family: system-ui, sans-serif;">
  <div style="font-weight: bold; color: #b8860b; margin-bottom: 8px;">
    ⚠️ PyRIT Upgrade Suggestion
  </div>
  <div style="margin-bottom: 6px; color: #333;">{description}</div>
  <div style="font-size: 0.9em; color: #666; margin-bottom: 8px;">
    Changed in PyRIT <b>v{version}</b>
  </div>
  <div style="background: #f5f5f5; border-radius: 4px; padding: 8px 12px;
              font-family: monospace; font-size: 0.9em; line-height: 1.6;">
    <div style="color: #c0392b;">- {example_old}</div>
    <div style="color: #27ae60;">+ {example_new}</div>
  </div>
</div>"""


def format_suggestion_plain(suggestion: UpgradeSuggestion) -> str:
    """Format a suggestion as plain text for terminal/console display."""
    rule = suggestion.rule
    return _PLAIN_TEMPLATE.format(
        description=rule.description,
        version=rule.version_introduced,
        example_old=rule.example_old,
        example_new=rule.example_new,
    )


def format_suggestion_html(suggestion: UpgradeSuggestion) -> str:
    """Format a suggestion as HTML for Jupyter notebook display."""
    rule = suggestion.rule
    return _HTML_TEMPLATE.format(
        description=rule.description,
        version=rule.version_introduced,
        example_old=rule.example_old,
        example_new=rule.example_new,
    )


# ---------------------------------------------------------------------------
# IPython integration
# ---------------------------------------------------------------------------

_is_enabled: bool = False


def _ipython_exception_handler(
    shell: Any,
    etype: type[BaseException],
    evalue: BaseException,
    tb: Any,
    tb_offset: Any = None,
) -> None:
    """Custom IPython exception handler that appends upgrade suggestions.

    This handler:
    1. Always shows the original traceback via IPython's default renderer
    2. Checks if the error matches any migration rules
    3. If matched, displays an upgrade suggestion below the traceback
    """
    # Step 1: Show the original traceback using IPython's default method
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    # Step 2: Try to match and display a suggestion
    try:
        if tb is not None:
            tb_lines = tb_module.format_tb(tb)
        else:
            tb_lines = []

        suggestion = suggest_fix(evalue, tb_lines)
        if suggestion is None:
            return

        # Display using rich HTML if available, plain text otherwise
        try:
            from IPython.display import HTML, display

            display(HTML(format_suggestion_html(suggestion)))
        except ImportError:
            print(format_suggestion_plain(suggestion))

    except Exception:
        # The helper must never interfere with normal error display
        logger.debug("Upgrade helper display failed", exc_info=True)


def enable_upgrade_helper() -> bool:
    """Enable the PyRIT notebook upgrade helper.

    Registers a custom IPython exception handler that shows upgrade
    suggestions when errors match known breaking changes. The original
    traceback is always displayed first.

    This function is idempotent — calling it multiple times has no
    additional effect.

    Returns:
        True if the helper was enabled, False if IPython is not available
        or the helper is already enabled.
    """
    global _is_enabled

    if _is_enabled:
        return True

    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        logger.info("Not in an IPython session; upgrade helper not enabled.")
        return False

    if ip is None:
        return False

    # Register our handler for common upgrade-related exception types
    ip.set_custom_exc(
        (ImportError, AttributeError, TypeError, ModuleNotFoundError),
        _ipython_exception_handler,
    )
    _is_enabled = True
    logger.info("PyRIT notebook upgrade helper enabled.")
    return True


def disable_upgrade_helper() -> bool:
    """Disable the PyRIT notebook upgrade helper.

    Removes the custom exception handler. This function is idempotent.

    Returns:
        True if the helper was disabled, False if it was not enabled
        or IPython is not available.
    """
    global _is_enabled

    if not _is_enabled:
        return False

    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        _is_enabled = False
        return False

    if ip is None:
        _is_enabled = False
        return False

    # Remove our custom handler by setting an empty tuple
    ip.set_custom_exc((), _ipython_exception_handler)
    _is_enabled = False
    logger.info("PyRIT notebook upgrade helper disabled.")
    return True


def is_upgrade_helper_enabled() -> bool:
    """Check whether the upgrade helper is currently active."""
    return _is_enabled
