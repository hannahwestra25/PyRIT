# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Registry owning scenario preset storage, validation, and version-checked resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrit.models.catalog.scenario_preset import PresetIssue, ScenarioPreset, StoredPreset
from pyrit.registry.registry import Registry
from pyrit.registry.registry_metadata import RegistryMetadata
from pyrit.registry.scenario_preset_storage import ScenarioPresetConflictError, ScenarioPresetStorage

if TYPE_CHECKING:
    from pyrit.registry.components.scenario_registry import ScenarioMetadata
    from pyrit.scenario.core.technique_tokens import TechniqueIssueKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioPresetMetadata(RegistryMetadata):
    """
    Metadata describing a buildable preset class.

    Placeholder for a catalog that is intentionally empty: a preset is a document, not a
    class this registry constructs. It carries only the common ``RegistryMetadata``
    fields.
    """


class ScenarioPresetRegistry(Registry["ScenarioPreset", ScenarioPresetMetadata]):
    """
    The single owner of scenario preset storage, validation, and resolution.

    A preset is inert JSON rather than a component class, so the buildable class catalog
    inherited from ``Registry`` stays empty and ``_discover`` registers nothing. What the
    base contributes, and what this registry relies on, is the shared singleton
    machinery every other registry is reached through.

    **Storage is read through on every call; this registry holds no presets.**

    The version guarding every write is ``sha256`` of the stored bytes, chosen so that an
    edit made by hand or by another tool invalidates it too. A version computed from
    cached bytes could only describe what this process last read or wrote. Multiple
    workers and replicas are a supported topology, so each process would cache
    independently. ``resolve_preset`` compares a client-supplied version against the
    stored one to confirm the operator is launching the configuration they reviewed;
    served from a cache that comparison tests the cache against itself. And
    ``get_preset_version`` must reach documents that cannot be parsed at all, which a map
    of parsed instances cannot hold.
    """

    def __init__(self, *, storage: ScenarioPresetStorage | None = None) -> None:
        """
        Initialize the registry without touching storage.

        Args:
            storage (ScenarioPresetStorage | None): Storage to read and write presets
                through. Defaults to the standard directory, constructed on first use so
                that merely importing the registry never creates a directory.
        """
        super().__init__()
        self._storage = storage

    def _discover(self) -> None:
        """Register no classes: a preset is a document, not a class this registry builds."""

    def _metadata_class(self) -> type[ScenarioPresetMetadata]:
        """Return ``ScenarioPresetMetadata``; unused while the buildable catalog is empty."""
        return ScenarioPresetMetadata

    @property
    def source(self) -> str:
        """Credential-free storage source presets are read from and written to."""
        return self._get_storage().display_source

    def configure_source(self, source: str | None) -> None:
        """
        Point the registry at a local directory or Azure Blob source.

        Args:
            source (str | None): The configured source, or None to use the default
                directory under the PyRIT configuration path.
        """
        self._storage = ScenarioPresetStorage(source=source)

    def scan_stored_presets(self) -> int:
        """
        Read every stored preset once so unusable documents are reported at startup.

        ``list_presets`` skips a document it cannot parse, so without this pass a broken
        preset stays invisible until someone opens the library and wonders where it went.

        The pass checks the schema only. Resolving a preset against this deployment, as
        ``check_preset`` does, needs scenario metadata, and building that instantiates
        every registered scenario class — which this early in boot has neither its
        environment nor its memory nor its techniques, so the pass would raise rather
        than merely be slow.

        Failures are logged rather than raised: unusable operator data must not stop the
        service that would let the operator repair it.

        Returns:
            int: The number of presets that parsed cleanly.
        """
        try:
            presets = self._get_storage().list_presets()
        except Exception:
            logger.exception("Could not read scenario presets at startup; the preset library may be unavailable.")
            return 0

        logger.info(f"Loaded {len(presets)} scenario preset(s) from {self.source}.")
        return len(presets)

    def list_presets(self) -> list[StoredPreset]:
        """
        Read every stored preset, skipping any that cannot be parsed.

        Returns:
            list[StoredPreset]: Stored presets and their versions, sorted by preset name.
        """
        stored = self._get_storage().list_presets()
        return [stored[name] for name in sorted(stored)]

    def get_preset(self, *, name: str) -> StoredPreset | None:
        """
        Read one stored preset together with the version needed to save an edit to it.

        Args:
            name (str): The preset name.

        Returns:
            StoredPreset | None: The preset and its version, or None if it is absent or
            malformed.

        Raises:
            ValueError: If *name* is not a legal preset name.
        """
        return self._get_storage().load_preset(name)

    def get_preset_version(self, *, name: str) -> str | None:
        """
        Read the version of one stored document without parsing it.

        This is the recovery path, not the ordinary way to read a version: a document
        that cannot be parsed is unreachable through ``get_preset``, so exposing its
        version lets a caller offer to overwrite the broken file instead of leaving the
        name permanently unusable.

        Args:
            name (str): The preset name.

        Returns:
            str | None: The document version, or None if no document is stored.

        Raises:
            ValueError: If *name* is not a legal preset name.
        """
        return self._get_storage().get_preset_version(name)

    def save_preset(self, *, preset: ScenarioPreset, expected_version: str | None) -> StoredPreset:
        """
        Persist one preset, refusing a write that would discard a concurrent edit.

        References this deployment cannot resolve are not rejected here. A preset
        authored against one deployment must remain savable on another, so unresolvable
        references are reported by ``check_preset`` instead.

        Args:
            preset (ScenarioPreset): The preset to persist.
            expected_version (str | None): None to create a preset that must not already
                exist, or the version returned when the edited preset was read.

        Returns:
            StoredPreset: The persisted preset and its new version.

        Raises:
            ScenarioPresetConflictError: If the stored version does not match *expected_version*.
            ValueError: If the preset name is not a legal preset name.
        """
        stored = self._get_storage().save_preset(preset=preset, expected_version=expected_version)
        logger.info(f"Saved scenario preset '{preset.name}'.")
        return stored

    def ensure_preset(self, *, preset: ScenarioPreset) -> bool:
        """
        Persist one preset only if nothing is stored under its name.

        This is how code contributes a preset — an initializer shipping a starting
        library, for instance. It is deliberately create-if-absent rather than a write:
        the stored document is the operator's, so a preset they have since edited, or
        deliberately replaced, is left alone.

        The create and the existence test are the same conditional write, so two workers
        starting together cannot both decide the name is free. Nothing is reported when
        the name is taken; the caller wanted a preset to exist, and one does.

        Args:
            preset (ScenarioPreset): The preset to persist if its name is free.

        Returns:
            bool: True if this call created the preset, False if one was already stored.

        Raises:
            ValueError: If the preset name is not a legal preset name.
        """
        try:
            self.save_preset(preset=preset, expected_version=None)
        except ScenarioPresetConflictError:
            logger.debug(f"Scenario preset '{preset.name}' already exists; leaving the stored preset alone.")
            return False
        return True

    def delete_preset(self, *, name: str) -> bool:
        """
        Delete one stored preset, reporting whether it was there to delete.

        The delete is unconditional by design. A version is what guards an edit from
        discarding someone else's concurrent edit; a delete has no such content to lose,
        and an operator removing a preset means it should be gone whatever it last said.

        The answer comes from the delete itself, so it stays correct under concurrency:
        when two callers race to remove the same preset, exactly one is told it removed
        it. An existence check followed by a delete would tell both that they did, and
        would report a preset created between the two steps as missing.

        Args:
            name (str): The preset name.

        Returns:
            bool: True if a document was stored under *name*, False if nothing was.

        Raises:
            ValueError: If *name* is not a legal preset name.
            TimeoutError: If a local delete could not acquire the document lock.
        """
        deleted = self._get_storage().delete_preset(name)
        if deleted:
            logger.info(f"Deleted scenario preset '{name}'.")
        return deleted

    def resolve_preset(self, *, name: str, expected_version: str | None) -> StoredPreset | None:
        """
        Read one preset for launch, confirming it is the version the caller reviewed.

        The comparison is against a fresh read rather than anything this process is
        holding. An edit landing between the preview and the launch is the case the
        check exists to catch, so the stored document is the only thing it can be
        compared against.

        Args:
            name (str): The preset name.
            expected_version (str | None): The version the caller read, or None to accept
                whatever is stored now.

        Returns:
            StoredPreset | None: The preset and its version, or None if nothing usable is
            stored under *name*.

        Raises:
            ScenarioPresetConflictError: If the stored version does not match *expected_version*.
            ValueError: If *name* is not a legal preset name.
        """
        stored = self._get_storage().load_preset(name)
        if stored is None:
            return None

        if expected_version is not None and expected_version != stored.version:
            raise ScenarioPresetConflictError(
                name=name, expected_version=expected_version, actual_version=stored.version
            )
        return stored

    def check_preset(self, *, preset: ScenarioPreset) -> list[PresetIssue]:
        """
        Check one preset against this deployment's registered scenarios and converters.

        This is the only validation path. A preset is resolved by the same code the
        launch path resolves it with, so a preset reported as runnable runs and a preset
        reported as broken is broken. Re-deriving any part of that check here would let
        the two answers drift apart, which is how an operator ends up being told a
        working preset is invalid.

        Results are advisory. Nothing here rejects a preset.

        Args:
            preset (ScenarioPreset): The preset to check.

        Returns:
            list[PresetIssue]: Advisory issues, empty when the preset resolves here.
        """
        from pyrit.registry.components.scenario_registry import ScenarioRegistry

        metadata = ScenarioRegistry.get_registry_singleton().get_registered_class_metadata(preset.scenario_name)
        if metadata is None:
            return [
                PresetIssue(
                    field="scenario_name",
                    message=f"Scenario '{preset.scenario_name}' is not registered in this deployment.",
                )
            ]

        return [
            *_technique_issues(preset=preset, metadata=metadata),
            *_parameter_issues(preset=preset, metadata=metadata),
            *_forbidden_baseline_issues(preset=preset, metadata=metadata),
        ]

    def _get_storage(self) -> ScenarioPresetStorage:
        """
        Return storage, constructing the default backend on first use.

        Returns:
            ScenarioPresetStorage: The configured storage backend.
        """
        if self._storage is None:
            self._storage = ScenarioPresetStorage()
        return self._storage


def _technique_issues(*, preset: ScenarioPreset, metadata: ScenarioMetadata) -> list[PresetIssue]:
    """
    Report technique references this deployment cannot resolve.

    The resolution is the launch path's own, called on the scenario's technique catalog
    rather than on the name tuples projected from it, so a preset reported as broken is
    one the launch path would reject and a preset reported as runnable runs. Re-deriving
    any part of it here is how an operator ends up being told a working preset is
    invalid.

    Issues are grouped by kind because an operator repairing a preset wants one line
    naming every unknown technique, not one line per token.

    Args:
        preset (ScenarioPreset): The preset to check.
        metadata (ScenarioMetadata): Registry metadata for the scenario it names.

    Returns:
        list[PresetIssue]: One issue per kind of unresolvable reference, or an empty list.
    """
    if not preset.techniques:
        return []

    from pyrit.scenario.core.technique_tokens import CONVERTER_MODIFIER_PREFIX, resolve_technique_tokens

    resolution = resolve_technique_tokens(
        tokens=preset.techniques,
        technique_class=metadata.technique_class,
        scenario_name=metadata.registry_name,
    )

    descriptions: tuple[tuple[TechniqueIssueKind, str], ...] = (
        ("unknown_technique", f"Scenario '{metadata.registry_name}' does not define"),
        ("unknown_converter", "This deployment has no registered converter named"),
        ("malformed_modifier", f"Technique modifiers must use the '{CONVERTER_MODIFIER_PREFIX}' prefix; got"),
    )

    issues: list[PresetIssue] = []
    for kind, description in descriptions:
        names = sorted({issue.name for issue in resolution.issues if issue.kind == kind})
        if names:
            issues.append(PresetIssue(field="techniques", message=f"{description}: {', '.join(names)}."))
    return issues


def _parameter_issues(*, preset: ScenarioPreset, metadata: ScenarioMetadata) -> list[PresetIssue]:
    """
    Report scenario parameters this scenario will not accept.

    This runs the resolution the launch path runs rather than comparing names against
    the declaration, because a declared parameter carrying a value it cannot be coerced
    to passes a name comparison and then fails the run — the operator is told the preset
    is fine and finds out otherwise at launch.

    Resolution stops at the first value it cannot coerce, so at most one coercion
    failure is reported per check; undeclared names are still reported together.

    Args:
        preset (ScenarioPreset): The preset to check.
        metadata (ScenarioMetadata): Registry metadata for the scenario it names.

    Returns:
        list[PresetIssue]: A single issue describing what will not resolve, or an empty list.
    """
    if not preset.scenario_params:
        return []

    from pyrit.registry.resolution import resolve_declared_params

    try:
        resolve_declared_params(
            declared=list(metadata.supported_parameters),
            raw_args=preset.scenario_params,
            owner=f"Scenario '{metadata.registry_name}'",
        )
    except ValueError as error:
        return [PresetIssue(field="scenario_params", message=str(error))]
    return []


def _forbidden_baseline_issues(*, preset: ScenarioPreset, metadata: ScenarioMetadata) -> list[PresetIssue]:
    """
    Report a baseline request the scenario forbids.

    Args:
        preset (ScenarioPreset): The preset to check.
        metadata (ScenarioMetadata): Registry metadata for the scenario it names.

    Returns:
        list[PresetIssue]: A single issue when the scenario forbids a requested baseline.
    """
    if not preset.include_baseline or metadata.baseline_policy != "forbidden":
        return []

    return [
        PresetIssue(
            field="include_baseline",
            message=f"Scenario '{metadata.registry_name}' does not support a baseline run.",
        )
    ]
