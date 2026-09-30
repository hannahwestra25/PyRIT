# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario preset storage."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyrit.exceptions.exception_classes import ScenarioPresetConflictError
from pyrit.models.catalog.scenario_preset import ScenarioPreset, ScenarioPresetProvenance
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


def test_local_storage_round_trips_a_preset(tmp_path: Path) -> None:
    """Test that a saved preset loads back with its fields intact."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    preset = _make_preset(
        techniques=["crescendo"],
        dataset_names=["harmbench"],
        max_dataset_size=25,
        dataset_filters={"harm_categories": ["violence"]},
        include_baseline=True,
        scenario_params={"max_turns": 3},
        description="Nightly smoke suite",
    )

    storage.save_preset(preset=preset, expected_version=None)
    loaded = storage.load_preset("nightly")

    assert loaded is not None
    assert loaded.techniques == ["crescendo"]
    assert loaded.dataset_names == ["harmbench"]
    assert loaded.max_dataset_size == 25
    assert loaded.dataset_filters == {"harm_categories": ["violence"]}
    assert loaded.include_baseline is True
    assert loaded.scenario_params == {"max_turns": 3}
    assert loaded.description == "Nightly smoke suite"


def test_unset_fields_round_trip_as_none_not_false(tmp_path: Path) -> None:
    """Test the tri-state regression: an unset override must not deserialize as a disabled one."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    storage.save_preset(preset=_make_preset(), expected_version=None)
    loaded = storage.load_preset("nightly")

    assert loaded is not None
    assert loaded.include_baseline is None
    assert loaded.max_dataset_size is None
    assert loaded.techniques is None

    stored = json.loads((tmp_path / "nightly.json").read_text(encoding="utf-8"))
    assert "include_baseline" not in stored
    assert "max_dataset_size" not in stored


def test_explicit_false_round_trips_as_false(tmp_path: Path) -> None:
    """Test that an explicitly disabled override survives storage."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    storage.save_preset(preset=_make_preset(include_baseline=False), expected_version=None)
    loaded = storage.load_preset("nightly")

    assert loaded is not None
    assert loaded.include_baseline is False


def test_create_assigns_version_one(tmp_path: Path) -> None:
    """Test that creating a preset starts its change counter at one."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    saved = storage.save_preset(preset=_make_preset(version=99), expected_version=None)

    assert saved.version == 1


def test_update_increments_version(tmp_path: Path) -> None:
    """Test that each accepted save advances the change counter."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    first = storage.save_preset(preset=_make_preset(), expected_version=None)

    second = storage.save_preset(preset=_make_preset(description="changed"), expected_version=first.version)
    third = storage.save_preset(preset=_make_preset(description="again"), expected_version=second.version)

    assert (first.version, second.version, third.version) == (1, 2, 3)


def test_stale_version_save_is_rejected(tmp_path: Path) -> None:
    """Test that a save based on a superseded version loses the concurrency check."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(), expected_version=None)
    storage.save_preset(preset=_make_preset(description="first writer"), expected_version=1)

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(description="second writer"), expected_version=1)

    assert error.value.expected_version == 1
    assert error.value.actual_version == 2
    assert error.value.status_code == 409


def test_stale_save_does_not_overwrite_the_winner(tmp_path: Path) -> None:
    """Test that a rejected save leaves stored content untouched."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(description="original"), expected_version=None)

    with pytest.raises(ScenarioPresetConflictError):
        storage.save_preset(preset=_make_preset(description="clobber"), expected_version=99)

    loaded = storage.load_preset("nightly")
    assert loaded is not None
    assert loaded.description == "original"


def test_create_over_existing_name_is_rejected(tmp_path: Path) -> None:
    """Test that creating a preset that already exists is a conflict, not an overwrite."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(), expected_version=None)

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(description="second"), expected_version=None)

    assert error.value.expected_version is None
    assert error.value.actual_version == 1


def test_update_of_missing_preset_is_rejected(tmp_path: Path) -> None:
    """Test that updating a preset deleted by someone else is a conflict."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(), expected_version=3)

    assert error.value.actual_version is None


def test_save_forces_user_provenance(tmp_path: Path) -> None:
    """Test that stored presets are always user-owned regardless of the submitted value."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    saved = storage.save_preset(preset=_make_preset(), expected_version=None)

    assert saved.provenance is ScenarioPresetProvenance.USER
    loaded = storage.load_preset("nightly")
    assert loaded is not None
    assert loaded.provenance is ScenarioPresetProvenance.USER


def test_builtin_preset_is_never_written(tmp_path: Path) -> None:
    """Test that a built-in preset cannot be persisted to user storage."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    builtin = _make_preset(provenance=ScenarioPresetProvenance.BUILT_IN)

    with pytest.raises(ValueError, match="built in and cannot be saved"):
        storage.save_preset(preset=builtin, expected_version=None)

    assert list(tmp_path.glob("*.json")) == []


def test_load_missing_preset_returns_none(tmp_path: Path) -> None:
    """Test that an absent preset loads as None rather than raising."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    assert storage.load_preset("absent") is None


def test_list_and_delete_presets(tmp_path: Path) -> None:
    """Test the storage lifecycle for multiple presets."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(name="nightly"), expected_version=None)
    storage.save_preset(preset=_make_preset(name="weekly"), expected_version=None)

    assert sorted(storage.list_presets()) == ["nightly", "weekly"]
    storage.delete_preset("nightly")
    assert sorted(storage.list_presets()) == ["weekly"]


def test_delete_missing_preset_is_silent(tmp_path: Path) -> None:
    """Test that deleting an absent preset is not an error."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    storage.delete_preset("absent")


def test_malformed_preset_is_skipped_not_fatal(tmp_path: Path) -> None:
    """Test that one unparseable file does not prevent the rest of the library from loading."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(name="valid"), expected_version=None)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "wrong_shape.json").write_text('["a list"]', encoding="utf-8")

    presets = storage.list_presets()

    assert sorted(presets) == ["valid"]
    assert storage.load_preset("broken") is None


def test_document_name_overrides_payload_name(tmp_path: Path) -> None:
    """Test that the file name is authoritative, so a load and its later save agree on the key."""
    (tmp_path / "actual_key.json").write_text(
        json.dumps({"name": "different", "scenario_name": "foundry.red_team_agent", "version": 1}),
        encoding="utf-8",
    )
    storage = ScenarioPresetStorage(source=str(tmp_path))

    loaded = storage.load_preset("actual_key")

    assert loaded is not None
    assert loaded.name == "actual_key"
    assert sorted(storage.list_presets()) == ["actual_key"]


def test_local_storage_returns_preset_path(tmp_path: Path) -> None:
    """Test resolving the displayed path for a local preset."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    assert storage.get_preset_source("nightly") == str(tmp_path / "nightly.json")
    assert storage.display_source == str(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "https://account.blob.attacker.example/presets",
        "https://user@account.blob.core.windows.net/presets",
        "https://blob.core.windows.net/presets",
    ],
)
def test_blob_storage_rejects_untrusted_authorities(source: str) -> None:
    """Test rejecting Blob lookalikes before Azure credentials are acquired."""
    with pytest.raises(ValueError, match="local directory or Azure Blob container URI"):
        ScenarioPresetStorage(source=source)


def test_blob_storage_round_trips_and_ignores_other_extensions() -> None:
    """Test container storage operations and that only JSON documents are listed."""
    document = json.dumps({"name": "nightly", "scenario_name": "foundry.red_team_agent", "version": 4})
    client = MagicMock()
    client.__enter__.return_value = client
    client.list_blobs.return_value = [
        SimpleNamespace(name="nightly.json"),
        SimpleNamespace(name="notes.txt"),
        SimpleNamespace(name="archive/ignored.json"),
    ]
    client.download_blob.return_value.readall.return_value = document.encode("utf-8")
    source = "https://account.blob.core.windows.net/presets?sp=rwd&sig=secret"
    storage = ScenarioPresetStorage(source=source)

    with patch("azure.storage.blob.ContainerClient.from_container_url", return_value=client):
        presets = storage.list_presets()
        saved = storage.save_preset(preset=_make_preset(description="updated"), expected_version=4)

    assert sorted(presets) == ["nightly"]
    assert presets["nightly"].version == 4
    assert saved.version == 5
    assert storage.display_source == "https://account.blob.core.windows.net/presets"
    assert client.upload_blob.call_args.kwargs["name"] == "nightly.json"


def test_blob_storage_reads_missing_document_as_none() -> None:
    """Test that a missing blob loads as None instead of propagating an Azure error."""
    from azure.core.exceptions import ResourceNotFoundError

    client = MagicMock()
    client.__enter__.return_value = client
    client.download_blob.side_effect = ResourceNotFoundError("missing")
    storage = ScenarioPresetStorage(source="https://account.blob.core.windows.net/presets?sig=secret")

    with patch("azure.storage.blob.ContainerClient.from_container_url", return_value=client):
        assert storage.load_preset("absent") is None
