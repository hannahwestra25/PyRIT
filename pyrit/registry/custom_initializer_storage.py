# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Storage backends for custom initializer Python source."""

from __future__ import annotations

from pyrit.registry.file_document_storage import FileDocumentStorage


class CustomInitializerStorage(FileDocumentStorage):
    """Read and write custom initializer scripts in a directory or blob container."""

    def __init__(self, *, source: str) -> None:
        """
        Initialize storage from a local directory or Azure Blob source URI.

        Raises:
            ValueError: If the source has an unsupported URI scheme.
        """
        super().__init__(source=source, extension=".py", source_label="Custom initializer")

    def get_script_source(self, name: str) -> str:
        """
        Get the credential-free location of a custom initializer script.

        Returns:
            str: Local file path or Azure Blob URI for the script.
        """
        return self._get_document_source(name)

    def list_scripts(self) -> dict[str, str]:
        """
        List stored Python scripts by registry name.

        Returns:
            dict[str, str]: Script content keyed by registry name.
        """
        return self._list_documents()

    def save_script(self, *, name: str, content: str) -> None:
        """Persist one custom initializer script."""
        self._save_document(name=name, content=content)

    def delete_script(self, name: str) -> None:
        """Delete one custom initializer script if it exists."""
        self._delete_document(name)
