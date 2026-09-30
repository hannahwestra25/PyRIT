# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Storage backends for user-authored scenario presets."""

from __future__ import annotations

import json
import logging

from pyrit.exceptions.exception_classes import ScenarioPresetConflictError
from pyrit.models.catalog.scenario_preset import ScenarioPreset, ScenarioPresetProvenance
from pyrit.registry.file_document_storage import FileDocumentStorage

logger = logging.getLogger(__name__)


class ScenarioPresetStorage(FileDocumentStorage):
    """
    Read and write user-authored scenario presets as JSON documents.

    Only user presets are stored. Built-in presets come from initializer code and are
    never written here, so a file on disk always means a user authored it.

    Writes are guarded by an optimistic-concurrency check on ``version``. The stored
    version is authoritative: a caller supplies the version it edited against, and the
    save is refused if storage has moved on.

    The check is read-then-write rather than a true compare-and-swap, so two saves racing
    within the same instant can both observe the same version and the later write wins.
    It is aimed at the realistic case — a person editing a copy that went stale minutes
    ago — not at concurrent writers. Closing that gap needs backend-specific conditional
    writes (blob ETags have no local-filesystem equivalent) and is deliberately deferred.
    """

    def __init__(self, *, source: str) -> None:
        """
        Initialize storage from a local directory or Azure Blob source URI.

        Raises:
            ValueError: If the source has an unsupported URI scheme.
        """
        super().__init__(source=source, extension=".json", source_label="Scenario preset")

    def get_preset_source(self, name: str) -> str:
        """
        Get the credential-free location of one stored preset.

        Returns:
            str: Local file path or Azure Blob URI for the preset.
        """
        return self._get_document_source(name)

    def list_presets(self) -> dict[str, ScenarioPreset]:
        """
        Read every stored preset, skipping any that cannot be parsed.

        A malformed or hand-edited file must not prevent the rest of the library from
        loading, so failures are logged and that preset is omitted.

        Returns:
            dict[str, ScenarioPreset]: Presets keyed by name.
        """
        presets: dict[str, ScenarioPreset] = {}
        for name, content in self._list_documents().items():
            preset = self._parse_preset(name=name, content=content)
            if preset is not None:
                presets[preset.name] = preset
        return presets

    def load_preset(self, name: str) -> ScenarioPreset | None:
        """
        Read one stored preset.

        Returns:
            ScenarioPreset | None: The preset, or ``None`` if it is absent or malformed.
        """
        content = self._read_document(name)
        if content is None:
            return None
        return self._parse_preset(name=name, content=content)

    def save_preset(self, *, preset: ScenarioPreset, expected_version: int | None) -> ScenarioPreset:
        """
        Persist one preset, assigning its version.

        The caller states its intent through *expected_version* rather than through the
        version on *preset*, so creating and updating are never ambiguous and a
        client-supplied version can never be trusted into storage.

        Args:
            preset (ScenarioPreset): The preset to persist.
            expected_version (int | None): ``None`` to create a preset that must not already
                exist, or the version the edit was based on.

        Returns:
            ScenarioPreset: The persisted preset, carrying its newly assigned version.

        Raises:
            ScenarioPresetConflictError: If the stored version does not match *expected_version*.
            ValueError: If *preset* is built-in, which is never stored.
        """
        if preset.is_builtin:
            raise ValueError(
                f"Scenario preset '{preset.name}' is built in and cannot be saved. "
                "Fork it under a new name to customize it."
            )

        existing = self.load_preset(preset.name)
        actual_version = existing.version if existing is not None else None
        if actual_version != expected_version:
            raise ScenarioPresetConflictError(
                name=preset.name, expected_version=expected_version, actual_version=actual_version
            )

        saved = preset.model_copy(
            update={
                "version": 1 if existing is None else existing.version + 1,
                "provenance": ScenarioPresetProvenance.USER,
            }
        )
        self._save_document(name=saved.name, content=self._serialize_preset(saved))
        return saved

    def delete_preset(self, name: str) -> None:
        """Delete one stored preset if it exists."""
        self._delete_document(name)

    @staticmethod
    def _serialize_preset(preset: ScenarioPreset) -> str:
        """
        Serialize a preset to stored JSON.

        Unset fields are omitted rather than written as ``null`` so a stored preset reads
        as the set of decisions its author actually made.

        Returns:
            str: JSON document content.
        """
        return json.dumps(preset.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _parse_preset(*, name: str, content: str) -> ScenarioPreset | None:
        """
        Parse one stored preset document.

        The document name is authoritative, overriding any ``name`` inside the payload.
        It is the storage key, so letting the payload disagree would make a load return a
        preset whose subsequent save wrote to a different file.

        Returns:
            ScenarioPreset | None: The parsed preset, or ``None`` if it is malformed.
        """
        try:
            payload = json.loads(content)
        except ValueError:
            logger.exception(f"Skipping stored scenario preset '{name}': it is not valid JSON.")
            return None

        if not isinstance(payload, dict):
            logger.error(f"Skipping stored scenario preset '{name}': it is not a JSON object.")
            return None

        fields = {**payload, "name": name, "provenance": ScenarioPresetProvenance.USER}
        try:
            return ScenarioPreset.model_validate(fields)
        except Exception:
            logger.exception(f"Skipping stored scenario preset '{name}': it is not a valid preset.")
            return None
