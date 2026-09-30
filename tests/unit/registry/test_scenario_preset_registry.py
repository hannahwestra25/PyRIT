# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenario preset registry."""

import logging
from pathlib import Path

import pytest

from pyrit.exceptions.exception_classes import ScenarioPresetConflictError
from pyrit.models.catalog.scenario_preset import ScenarioPreset, ScenarioPresetProvenance
from pyrit.registry.scenario_preset_registry import ScenarioPresetRegistry
from pyrit.registry.scenario_preset_storage import ScenarioPresetStorage


def _make_preset(**overrides: object) -> ScenarioPreset:
    """
    Build a preset with test defaults.

    Returns:
        ScenarioPreset: The constructed preset.
    """
    fields: dict[str, object] = {"name": "nightly", "scenario_name": "foundry.red_team_agent"}
    fields.update(overrides)
    return ScenarioPreset(**fields)  # type: ignore[arg-type]


@pytest.fixture
def registry(tmp_path: Path) -> ScenarioPresetRegistry:
    """
    Build a registry backed by an isolated local directory.

    Returns:
        ScenarioPresetRegistry: The registry under test.
    """
    return ScenarioPresetRegistry(storage=ScenarioPresetStorage(source=str(tmp_path)))


def test_list_presets_returns_union_sorted_by_name(registry: ScenarioPresetRegistry) -> None:
    """Test that the registry exposes code-registered and file-loaded presets together."""
    registry.register_builtin(_make_preset(name="builtin_suite"))
    registry.save_preset(preset=_make_preset(name="user_suite"), expected_version=None)

    names = [preset.name for preset in registry.list_presets()]

    assert names == ["builtin_suite", "user_suite"]


def test_register_builtin_forces_builtin_provenance(registry: ScenarioPresetRegistry) -> None:
    """Test that a registered built-in cannot be marked editable by its author."""
    registered = registry.register_builtin(_make_preset(provenance=ScenarioPresetProvenance.USER))

    assert registered.provenance is ScenarioPresetProvenance.BUILT_IN
    assert registry.is_builtin("nightly") is True


def test_register_builtin_rejects_duplicate_name(registry: ScenarioPresetRegistry) -> None:
    """Test that two built-ins cannot claim the same name."""
    registry.register_builtin(_make_preset())

    with pytest.raises(ValueError, match="already registered"):
        registry.register_builtin(_make_preset())


def test_get_preset_resolves_both_provenances(registry: ScenarioPresetRegistry) -> None:
    """Test name resolution across both sources."""
    registry.register_builtin(_make_preset(name="builtin_suite"))
    registry.save_preset(preset=_make_preset(name="user_suite"), expected_version=None)

    assert registry.get_preset("builtin_suite") is not None
    assert registry.get_preset("user_suite") is not None
    assert registry.get_preset("absent") is None


def test_saving_over_a_builtin_name_is_rejected(registry: ScenarioPresetRegistry) -> None:
    """Test that forking a built-in requires a new name."""
    registry.register_builtin(_make_preset())

    with pytest.raises(ValueError, match="built in and cannot be saved"):
        registry.save_preset(preset=_make_preset(), expected_version=None)


def test_deleting_a_builtin_is_rejected(registry: ScenarioPresetRegistry) -> None:
    """Test that built-in presets cannot be removed."""
    registry.register_builtin(_make_preset())

    with pytest.raises(ValueError, match="built in and cannot be deleted"):
        registry.delete_preset("nightly")


def test_fork_under_a_new_name_succeeds(registry: ScenarioPresetRegistry) -> None:
    """Test that the supported customization path works."""
    registry.register_builtin(_make_preset(name="builtin_suite", max_dataset_size=200))

    forked = registry.save_preset(
        preset=_make_preset(name="builtin_suite_fork", max_dataset_size=20), expected_version=None
    )

    assert forked.name == "builtin_suite_fork"
    assert forked.max_dataset_size == 20
    assert registry.get_preset("builtin_suite") is not None


def test_builtin_wins_collision_and_user_preset_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test the upgrade case: a new built-in shadows a stored preset without failing startup."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(description="user version"), expected_version=None)

    registry = ScenarioPresetRegistry(storage=storage)
    registry.register_builtin(_make_preset(description="shipped version"))

    with caplog.at_level(logging.WARNING):
        registry.load_stored_presets()

    resolved = registry.get_preset("nightly")
    assert resolved is not None
    assert resolved.description == "shipped version"
    assert resolved.is_builtin is True
    assert [preset.name for preset in registry.list_presets()] == ["nightly"]
    assert "a built-in preset already uses that name" in caplog.text


def test_load_stored_presets_keeps_non_colliding_presets(tmp_path: Path) -> None:
    """Test that a collision skips only the colliding preset."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(name="nightly"), expected_version=None)
    storage.save_preset(preset=_make_preset(name="weekly"), expected_version=None)

    registry = ScenarioPresetRegistry(storage=storage)
    registry.register_builtin(_make_preset(name="nightly"))
    registry.load_stored_presets()

    assert [preset.name for preset in registry.list_presets()] == ["nightly", "weekly"]
    weekly = registry.get_preset("weekly")
    assert weekly is not None
    assert weekly.is_builtin is False


def test_load_stored_presets_replaces_prior_state(tmp_path: Path) -> None:
    """Test that reloading reflects presets deleted outside the registry."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    registry = ScenarioPresetRegistry(storage=storage)
    registry.save_preset(preset=_make_preset(), expected_version=None)

    storage.delete_preset("nightly")
    registry.load_stored_presets()

    assert registry.get_preset("nightly") is None


def test_builtin_presets_are_never_written_to_storage(tmp_path: Path) -> None:
    """Test that registering a built-in does not persist anything."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    registry = ScenarioPresetRegistry(storage=storage)

    registry.register_builtin(_make_preset())

    assert storage.list_presets() == {}
    assert list(tmp_path.glob("*.json")) == []


def test_save_preset_refreshes_the_in_memory_copy(registry: ScenarioPresetRegistry) -> None:
    """Test that a save is visible through the registry without reloading."""
    first = registry.save_preset(preset=_make_preset(description="first"), expected_version=None)
    registry.save_preset(preset=_make_preset(description="second"), expected_version=first.version)

    resolved = registry.get_preset("nightly")

    assert resolved is not None
    assert resolved.description == "second"
    assert resolved.version == 2


def test_stale_save_propagates_conflict(registry: ScenarioPresetRegistry) -> None:
    """Test that the registry surfaces the storage concurrency check."""
    registry.save_preset(preset=_make_preset(), expected_version=None)

    with pytest.raises(ScenarioPresetConflictError):
        registry.save_preset(preset=_make_preset(description="stale"), expected_version=99)


def test_delete_preset_removes_from_registry_and_storage(tmp_path: Path) -> None:
    """Test that deletion clears both the cache and the backing file."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    registry = ScenarioPresetRegistry(storage=storage)
    registry.save_preset(preset=_make_preset(), expected_version=None)

    registry.delete_preset("nightly")

    assert registry.get_preset("nightly") is None
    assert storage.list_presets() == {}


def test_configure_storage_source_switches_backend(tmp_path: Path) -> None:
    """Test that the storage source can be pointed at an explicit directory."""
    registry = ScenarioPresetRegistry()
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    registry.configure_storage_source(str(presets_dir))
    registry.save_preset(preset=_make_preset(), expected_version=None)

    assert (presets_dir / "nightly.json").is_file()
