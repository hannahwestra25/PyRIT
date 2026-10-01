# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Registry unioning built-in and user-authored scenario presets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyrit.models.catalog.scenario_preset import ScenarioPreset, ScenarioPresetProvenance, StoredPreset
from pyrit.registry.scenario_preset_storage import ScenarioPresetStorage

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ScenarioPresetRegistry:
    """
    The union of built-in and user-authored scenario presets.

    Presets arrive from two places. Built-in presets are registered from initializer code
    at startup and are read-only, so they are held in memory. User presets are read from
    storage on each call rather than cached, because the same directory or blob container
    is routinely shared between a notebook, the API, and a second process; a cache would
    serve edits those callers can no longer see. The registry exists because only it sees
    both sources, so only it can resolve a name across them.

    Built-in presets win on a name collision. A user preset that collides is skipped with a
    warning rather than failing startup, because the collision arises when PyRIT ships a new
    built-in under a name a user already took, and an upgrade must not break a running
    install. To customize a built-in, fork it under a new name.
    """

    def __init__(self, *, storage: ScenarioPresetStorage | None = None) -> None:
        """
        Initialize the registry.

        Args:
            storage (ScenarioPresetStorage | None): Storage for user presets. Defaults to a
                local directory under the PyRIT configuration path.
        """
        self._storage = storage
        self._builtin_presets: dict[str, ScenarioPreset] = {}

    def configure_storage_source(self, source: str | None) -> None:
        """Configure the local directory or Azure Blob source for user presets."""
        self._storage = ScenarioPresetStorage(source=source or str(self._get_default_storage_dir()))

    def register_builtin(self, preset: ScenarioPreset) -> ScenarioPreset:
        """
        Register a preset that ships with PyRIT.

        Args:
            preset (ScenarioPreset): The preset to register. Its provenance is forced to
                built-in so a caller cannot register an editable preset by accident.

        Returns:
            ScenarioPreset: The registered preset.

        Raises:
            ValueError: If a built-in preset with the same name is already registered.
        """
        if preset.name in self._builtin_presets:
            raise ValueError(f"Built-in scenario preset '{preset.name}' is already registered.")

        registered = preset.model_copy(update={"provenance": ScenarioPresetProvenance.BUILT_IN})
        self._builtin_presets[registered.name] = registered
        return registered

    def get_preset(self, name: str) -> ScenarioPreset | None:
        """
        Resolve one preset by name.

        Returns:
            ScenarioPreset | None: The preset, or ``None`` if no preset uses that name.

        Raises:
            ValueError: If *name* is not a legal preset name.
        """
        builtin = self._builtin_presets.get(name)
        if builtin is not None:
            return builtin

        stored = self._get_storage().load_preset(name)
        return stored.preset if stored is not None else None

    def get_stored_preset(self, name: str) -> StoredPreset | None:
        """
        Read one user preset together with the version needed to save an edit to it.

        Returns:
            StoredPreset | None: The stored preset, or ``None`` if *name* is built in or
            no stored preset uses that name.

        Raises:
            ValueError: If *name* is not a legal preset name.
        """
        if name in self._builtin_presets:
            return None
        return self._get_storage().load_preset(name)

    def list_presets(self) -> list[ScenarioPreset]:
        """
        List every known preset.

        Returns:
            list[ScenarioPreset]: Built-in and user presets, sorted by name.
        """
        merged: dict[str, ScenarioPreset] = {}
        for name, stored in self._get_storage().list_presets().items():
            if name in self._builtin_presets:
                logger.warning(
                    f"Skipping stored scenario preset '{name}': a built-in preset already uses that name. "
                    "Rename the stored preset to keep using it."
                )
                continue
            merged[name] = stored.preset

        merged.update(self._builtin_presets)
        return [merged[name] for name in sorted(merged)]

    def is_builtin(self, name: str) -> bool:
        """Return whether *name* belongs to a built-in preset."""
        return name in self._builtin_presets

    def save_preset(self, *, preset: ScenarioPreset, expected_version: str | None) -> StoredPreset:
        """
        Persist a user preset.

        Args:
            preset (ScenarioPreset): The preset to persist.
            expected_version (str | None): ``None`` to create a preset that must not already
                exist, or the version returned when the edited preset was read.

        Returns:
            StoredPreset: The persisted preset and its new version.

        Raises:
            ScenarioPresetConflictError: If the stored version does not match *expected_version*.
            ValueError: If the name belongs to a built-in preset or is not a legal preset name.
        """
        self._reject_builtin(preset.name, action="saved")
        return self._get_storage().save_preset(preset=preset, expected_version=expected_version)

    def delete_preset(self, name: str) -> None:
        """
        Delete a user preset from storage.

        Raises:
            ValueError: If *name* belongs to a built-in preset or is not a legal preset name.
        """
        self._reject_builtin(name, action="deleted")
        self._get_storage().delete_preset(name)

    def _reject_builtin(self, name: str, *, action: str) -> None:
        """
        Refuse to mutate a built-in preset.

        Raises:
            ValueError: If *name* belongs to a built-in preset.
        """
        if name in self._builtin_presets:
            raise ValueError(
                f"Scenario preset '{name}' is built in and cannot be {action}. "
                "Fork it under a new name to customize it."
            )

    def _get_storage(self) -> ScenarioPresetStorage:
        """
        Return storage for user presets, creating the default backend on first use.

        Returns:
            ScenarioPresetStorage: The configured storage backend.
        """
        if self._storage is None:
            self._storage = ScenarioPresetStorage(source=str(self._get_default_storage_dir()))
        return self._storage

    @staticmethod
    def _get_default_storage_dir() -> Path:
        """
        Get the directory for storing user-authored presets.

        Returns:
            Path: Path to ``~/.pyrit/scenario_presets/``, created if needed.
        """
        # Deferred: importing pyrit.common.path triggers pyrit __init__.py
        from pyrit.common.path import CONFIGURATION_DIRECTORY_PATH

        presets_dir = CONFIGURATION_DIRECTORY_PATH / "scenario_presets"
        presets_dir.mkdir(parents=True, exist_ok=True)
        return presets_dir
