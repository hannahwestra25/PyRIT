# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario preset storage."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models.catalog.scenario_preset import ScenarioPreset, ScenarioPresetProvenance
from pyrit.registry.scenario_preset_storage import ScenarioPresetConflictError, ScenarioPresetStorage


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
    assert loaded.preset.techniques == ["crescendo"]
    assert loaded.preset.dataset_names == ["harmbench"]
    assert loaded.preset.max_dataset_size == 25
    assert loaded.preset.dataset_filters == {"harm_categories": ["violence"]}
    assert loaded.preset.include_baseline is True
    assert loaded.preset.scenario_params == {"max_turns": 3}
    assert loaded.preset.description == "Nightly smoke suite"


def test_unset_fields_round_trip_as_none_not_false(tmp_path: Path) -> None:
    """Test the tri-state regression: an unset override must not deserialize as a disabled one."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    storage.save_preset(preset=_make_preset(), expected_version=None)
    loaded = storage.load_preset("nightly")

    assert loaded is not None
    assert loaded.preset.include_baseline is None
    assert loaded.preset.max_dataset_size is None
    assert loaded.preset.techniques is None

    stored = json.loads((tmp_path / "nightly.json").read_text(encoding="utf-8"))
    assert "include_baseline" not in stored
    assert "max_dataset_size" not in stored


def test_explicit_false_round_trips_as_false(tmp_path: Path) -> None:
    """Test that an explicitly disabled override survives storage."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    storage.save_preset(preset=_make_preset(include_baseline=False), expected_version=None)
    loaded = storage.load_preset("nightly")

    assert loaded is not None
    assert loaded.preset.include_baseline is False


def test_version_is_not_stored_inside_the_document(tmp_path: Path) -> None:
    """Test that the conflict token lives beside the document, never inside the file it guards."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    saved = storage.save_preset(preset=_make_preset(), expected_version=None)

    assert saved.version
    assert "version" not in json.loads((tmp_path / "nightly.json").read_text(encoding="utf-8"))


def test_each_save_produces_a_new_version(tmp_path: Path) -> None:
    """Test that an accepted save supersedes the token the caller passed in."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    first = storage.save_preset(preset=_make_preset(), expected_version=None)

    second = storage.save_preset(preset=_make_preset(description="changed"), expected_version=first.version)

    assert second.version != first.version
    reloaded = storage.load_preset("nightly")
    assert reloaded is not None
    assert reloaded.version == second.version


def test_identical_content_keeps_the_same_version(tmp_path: Path) -> None:
    """Test that the token describes stored content, so a no-op save does not invalidate other readers."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    first = storage.save_preset(preset=_make_preset(), expected_version=None)

    second = storage.save_preset(preset=_make_preset(), expected_version=first.version)

    assert second.version == first.version


def test_stale_version_save_is_rejected(tmp_path: Path) -> None:
    """Test that a save based on a superseded version loses the concurrency check."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    created = storage.save_preset(preset=_make_preset(), expected_version=None)
    winner = storage.save_preset(preset=_make_preset(description="first writer"), expected_version=created.version)

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(description="second writer"), expected_version=created.version)

    assert error.value.expected_version == created.version
    assert error.value.actual_version == winner.version


def test_out_of_band_edit_invalidates_the_version(tmp_path: Path) -> None:
    """Test that a file edited outside this class is detected, not silently overwritten."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    created = storage.save_preset(preset=_make_preset(description="original"), expected_version=None)
    (tmp_path / "nightly.json").write_text(
        json.dumps({"scenario_name": "foundry.red_team_agent", "description": "edited by hand"}),
        encoding="utf-8",
    )

    with pytest.raises(ScenarioPresetConflictError):
        storage.save_preset(preset=_make_preset(description="stale client"), expected_version=created.version)

    loaded = storage.load_preset("nightly")
    assert loaded is not None
    assert loaded.preset.description == "edited by hand"


def test_stale_save_does_not_overwrite_the_winner(tmp_path: Path) -> None:
    """Test that a rejected save leaves stored content untouched."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    storage.save_preset(preset=_make_preset(description="original"), expected_version=None)

    with pytest.raises(ScenarioPresetConflictError):
        storage.save_preset(preset=_make_preset(description="clobber"), expected_version="not_the_stored_version")

    loaded = storage.load_preset("nightly")
    assert loaded is not None
    assert loaded.preset.description == "original"


def test_create_over_existing_name_is_rejected(tmp_path: Path) -> None:
    """Test that creating a preset that already exists is a conflict, not an overwrite."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    created = storage.save_preset(preset=_make_preset(), expected_version=None)

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(description="second"), expected_version=None)

    assert error.value.expected_version is None
    assert error.value.actual_version == created.version


def test_create_over_malformed_file_is_rejected(tmp_path: Path) -> None:
    """Test that an unreadable file still blocks a create, so hand-written content is not discarded."""
    storage = ScenarioPresetStorage(source=str(tmp_path))
    (tmp_path / "nightly.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ScenarioPresetConflictError):
        storage.save_preset(preset=_make_preset(), expected_version=None)

    assert (tmp_path / "nightly.json").read_text(encoding="utf-8") == "{not json"


def test_update_of_missing_preset_is_rejected(tmp_path: Path) -> None:
    """Test that updating a preset deleted by someone else is a conflict."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    with pytest.raises(ScenarioPresetConflictError) as error:
        storage.save_preset(preset=_make_preset(), expected_version="some_version")

    assert error.value.actual_version is None


def test_save_forces_user_provenance(tmp_path: Path) -> None:
    """Test that stored presets are always user-owned regardless of the submitted value."""
    storage = ScenarioPresetStorage(source=str(tmp_path))

    saved = storage.save_preset(
        preset=_make_preset(provenance=ScenarioPresetProvenance.BUILT_IN), expected_version=None
    )

    assert saved.preset.provenance is ScenarioPresetProvenance.USER
    loaded = storage.load_preset("nightly")
    assert loaded is not None
    assert loaded.preset.provenance is ScenarioPresetProvenance.USER


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


@pytest.mark.parametrize("name", ["../victim", "..\\victim", "nested/victim", "Nightly", "night-ly", ""])
def test_document_operations_reject_illegal_names(tmp_path: Path, name: str) -> None:
    """Test that a name which would escape the configured source is refused on every path."""
    source = tmp_path / "presets"
    source.mkdir()
    victim = tmp_path / "victim.json"
    victim.write_text("do not touch", encoding="utf-8")
    storage = ScenarioPresetStorage(source=str(source))

    with pytest.raises(ValueError, match="Invalid registry name"):
        storage.load_preset(name)
    with pytest.raises(ValueError, match="Invalid registry name"):
        storage.delete_preset(name)
    with pytest.raises(ValueError, match="Invalid registry name"):
        storage.get_preset_source(name)

    assert victim.read_text(encoding="utf-8") == "do not touch"


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
        json.dumps({"name": "different", "scenario_name": "foundry.red_team_agent"}),
        encoding="utf-8",
    )
    storage = ScenarioPresetStorage(source=str(tmp_path))

    loaded = storage.load_preset("actual_key")

    assert loaded is not None
    assert loaded.preset.name == "actual_key"
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
    document = json.dumps({"name": "nightly", "scenario_name": "foundry.red_team_agent"})
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
        saved = storage.save_preset(
            preset=_make_preset(description="updated"), expected_version=presets["nightly"].version
        )

    assert sorted(presets) == ["nightly"]
    assert saved.version != presets["nightly"].version
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
