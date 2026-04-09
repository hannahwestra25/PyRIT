# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.target_requirements import TargetRequirements


@pytest.fixture
def adapt_all_policy():
    return CapabilityHandlingPolicy(
        behaviors={
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.ADAPT,
            CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.JSON_OUTPUT: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.MULTI_MESSAGE_PIECES: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.EDITABLE_HISTORY: UnsupportedCapabilityBehavior.RAISE,
        }
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_default_has_empty_capabilities():
    reqs = TargetRequirements()
    assert reqs.required_capabilities == frozenset()


def test_init_with_capabilities():
    reqs = TargetRequirements(
        required_capabilities=frozenset({CapabilityName.MULTI_TURN, CapabilityName.SYSTEM_PROMPT})
    )
    assert CapabilityName.MULTI_TURN in reqs.required_capabilities
    assert CapabilityName.SYSTEM_PROMPT in reqs.required_capabilities


def test_frozen_dataclass_is_immutable():
    reqs = TargetRequirements()
    with pytest.raises(AttributeError):
        reqs.required_capabilities = frozenset({CapabilityName.MULTI_TURN})


# ---------------------------------------------------------------------------
# validate — all pass
# ---------------------------------------------------------------------------


def test_validate_passes_when_target_supports_all_natively():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps)
    reqs = TargetRequirements(
        required_capabilities=frozenset({CapabilityName.MULTI_TURN, CapabilityName.SYSTEM_PROMPT})
    )
    reqs.validate(configuration=config)


def test_validate_passes_when_policy_is_adapt(adapt_all_policy):
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=adapt_all_policy)
    reqs = TargetRequirements(
        required_capabilities=frozenset({CapabilityName.MULTI_TURN, CapabilityName.SYSTEM_PROMPT})
    )
    reqs.validate(configuration=config)


def test_validate_passes_with_empty_requirements():
    caps = TargetCapabilities(supports_multi_turn=True, supports_system_prompt=True)
    config = TargetConfiguration(capabilities=caps)
    reqs = TargetRequirements()
    reqs.validate(configuration=config)


# ---------------------------------------------------------------------------
# validate — failures
# ---------------------------------------------------------------------------


def test_validate_raises_when_capability_missing_and_policy_raise(adapt_all_policy):
    # Build with ADAPT so construction succeeds, then override policy for validate.
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=True)
    raise_policy = CapabilityHandlingPolicy(
        behaviors={
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
        }
    )
    # multi_turn is missing + RAISE → pipeline construction raises, so build with ADAPT first
    config = TargetConfiguration(capabilities=caps, policy=adapt_all_policy)
    # Swap in a RAISE policy to test validate behavior
    config._policy = raise_policy
    reqs = TargetRequirements(required_capabilities=frozenset({CapabilityName.MULTI_TURN}))
    with pytest.raises(ValueError, match="supports_multi_turn"):
        reqs.validate(configuration=config)


def test_validate_raises_for_non_normalizable_capability(adapt_all_policy):
    caps = TargetCapabilities(supports_editable_history=False)
    config = TargetConfiguration(capabilities=caps, policy=adapt_all_policy)
    reqs = TargetRequirements(required_capabilities=frozenset({CapabilityName.EDITABLE_HISTORY}))
    with pytest.raises(ValueError, match="supports_editable_history"):
        reqs.validate(configuration=config)


def test_validate_raises_on_first_unsatisfied_capability():
    """When multiple capabilities are missing, validate raises on the first (sorted) one."""
    # Both missing, both ADAPT → construction OK, then swap to RAISE to test validate
    adapt_policy = CapabilityHandlingPolicy(
        behaviors={
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.ADAPT,
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.ADAPT,
        }
    )
    caps = TargetCapabilities(supports_multi_turn=False, supports_system_prompt=False)
    config = TargetConfiguration(capabilities=caps, policy=adapt_policy)
    raise_policy = CapabilityHandlingPolicy(
        behaviors={
            CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
            CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
        }
    )
    config._policy = raise_policy
    reqs = TargetRequirements(
        required_capabilities=frozenset({CapabilityName.MULTI_TURN, CapabilityName.SYSTEM_PROMPT})
    )
    with pytest.raises(ValueError):
        reqs.validate(configuration=config)


def test_validate_mixed_adapt_and_raise(adapt_all_policy):
    """One capability adapts but another raises — validate should raise."""
    caps = TargetCapabilities(
        supports_multi_turn=False, supports_system_prompt=False, supports_json_output=False
    )
    config = TargetConfiguration(capabilities=caps, policy=adapt_all_policy)
    # multi_turn and system_prompt => ADAPT (OK), json_output => RAISE (fail)
    reqs = TargetRequirements(
        required_capabilities=frozenset(
            {CapabilityName.MULTI_TURN, CapabilityName.SYSTEM_PROMPT, CapabilityName.JSON_OUTPUT}
        )
    )
    with pytest.raises(ValueError, match="supports_json_output"):
        reqs.validate(configuration=config)
