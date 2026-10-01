# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared local-directory and Azure Blob storage for flat, named documents."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from pyrit.common.azure_storage import has_sas_signature, is_azure_blob_uri, redact_url_credentials
from pyrit.models.identifiers.class_name_utils import validate_registry_name

if TYPE_CHECKING:
    from collections.abc import Generator

    from azure.storage.blob import ContainerClient


class FileDocumentStorage:
    """
    Read and write flat, named text documents in a directory or blob container.

    Documents live directly under the configured source, one file per name, with a
    fixed extension. Nested paths are ignored so a virtual directory prefix behaves
    the same way in both backends.

    Every operation that addresses a single document validates the name first. The
    name becomes a path component in both backends, so an unvalidated name would let
    a caller read, overwrite, or delete a file outside the configured source. The
    check lives here rather than in each subclass so no document API can omit it.

    Subclasses supply the extension and a human-readable label for error messages,
    then expose a domain-specific API over the protected document operations.
    """

    def __init__(self, *, source: str, extension: str, source_label: str) -> None:
        """
        Initialize storage from a local directory or Azure Blob source URI.

        Args:
            source (str): Local directory path or Azure Blob container URI with an
                optional blob prefix.
            extension (str): File extension including the leading dot, such as ``".py"``.
            source_label (str): Human-readable label naming the stored documents, used
                in error messages.

        Raises:
            ValueError: If the source has an unsupported URI scheme.
        """
        self._source = source
        self._extension = extension
        self._is_blob = is_azure_blob_uri(source)
        if not self._is_blob and urlparse(source).scheme and not Path(source).drive:
            raise ValueError(
                f"{source_label} source must be a local directory or Azure Blob container URI "
                "with an optional blob prefix"
            )
        self._container_url, self._blob_prefix = self._parse_blob_source() if self._is_blob else (None, "")

    @property
    def display_source(self) -> str:
        """Storage source without Azure Blob credentials."""
        if not self._is_blob:
            return self._source
        return redact_url_credentials(self._source)

    def _get_document_source(self, name: str) -> str:
        """
        Get the credential-free location of one document.

        Returns:
            str: Local file path or Azure Blob URI for the document.

        Raises:
            ValueError: If *name* is not a legal registry name.
        """
        validate_registry_name(name)
        if self._is_blob:
            return f"{self.display_source.rstrip('/')}/{name}{self._extension}"
        return str(self._local_directory() / f"{name}{self._extension}")

    def _list_documents(self) -> dict[str, str]:
        """
        Read every stored document.

        Returns:
            dict[str, str]: Document content keyed by name.
        """
        if self._is_blob:
            return self._list_blob_documents()

        directory = self._local_directory(create=True)
        return {path.stem: path.read_text(encoding="utf-8") for path in sorted(directory.glob(f"*{self._extension}"))}

    def _read_document(self, name: str) -> str | None:
        """
        Read one document.

        Returns:
            str | None: Document content, or ``None`` if it does not exist.

        Raises:
            ValueError: If *name* is not a legal registry name.
        """
        validate_registry_name(name)
        if self._is_blob:
            from azure.core.exceptions import ResourceNotFoundError

            with self._open_container_client() as client:
                try:
                    return client.download_blob(self._get_blob_name(name)).readall().decode("utf-8")
                except ResourceNotFoundError:
                    return None

        path = self._local_directory() / f"{name}{self._extension}"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def _save_document(self, *, name: str, content: str) -> None:
        """
        Persist one document, overwriting any existing content.

        Raises:
            ValueError: If *name* is not a legal registry name.
        """
        validate_registry_name(name)
        if self._is_blob:
            with self._open_container_client() as client:
                client.upload_blob(name=self._get_blob_name(name), data=content.encode("utf-8"), overwrite=True)
        else:
            directory = self._local_directory(create=True)
            (directory / f"{name}{self._extension}").write_text(content, encoding="utf-8")

    def _delete_document(self, name: str) -> None:
        """
        Delete one document if it exists.

        Raises:
            ValueError: If *name* is not a legal registry name.
        """
        validate_registry_name(name)
        if self._is_blob:
            from azure.core.exceptions import ResourceNotFoundError

            with self._open_container_client() as client:
                with suppress(ResourceNotFoundError):
                    client.delete_blob(self._get_blob_name(name))
        else:
            (self._local_directory() / f"{name}{self._extension}").unlink(missing_ok=True)

    def _local_directory(self, *, create: bool = False) -> Path:
        """
        Resolve the configured local directory.

        Returns:
            Path: The expanded directory path.
        """
        directory = Path(self._source).expanduser()
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _list_blob_documents(self) -> dict[str, str]:
        """
        Read documents from the configured Azure Blob container.

        Returns:
            dict[str, str]: Document content keyed by blob stem.
        """
        documents: dict[str, str] = {}
        with self._open_container_client() as client:
            prefix = f"{self._blob_prefix}/" if self._blob_prefix else None
            blobs = client.list_blobs(name_starts_with=prefix) if prefix else client.list_blobs()
            blob_names = sorted(
                blob.name for blob in blobs if self._is_direct_document_blob(blob_name=blob.name, prefix=prefix)
            )
            for blob_name in blob_names:
                relative_name = blob_name.removeprefix(prefix or "")
                documents[PurePosixPath(relative_name).stem] = client.download_blob(blob_name).readall().decode("utf-8")
        return documents

    def _parse_blob_source(self) -> tuple[str, str]:
        """
        Split the configured source into a container URL and blob prefix.

        Returns:
            tuple[str, str]: The container URL and decoded blob prefix.
        """
        parsed_uri = urlparse(self._source)
        container_path, _, prefix = parsed_uri.path.strip("/").partition("/")
        container_url = parsed_uri._replace(path=f"/{container_path}", fragment="").geturl()
        return container_url, unquote(prefix).strip("/")

    def _get_blob_name(self, name: str) -> str:
        """
        Build the blob name for one document.

        Returns:
            str: The prefixed blob name.
        """
        file_name = f"{name}{self._extension}"
        return f"{self._blob_prefix}/{file_name}" if self._blob_prefix else file_name

    def _is_direct_document_blob(self, *, blob_name: str, prefix: str | None) -> bool:
        """Return whether a blob is a direct child of the configured prefix with the expected extension."""
        if prefix and not blob_name.startswith(prefix):
            return False
        relative_name = blob_name.removeprefix(prefix or "")
        return "/" not in relative_name and PurePosixPath(relative_name).suffix == self._extension

    @contextmanager
    def _open_container_client(self) -> Generator[ContainerClient, None, None]:
        """
        Yield an Azure Blob container client and close its credential.

        Yields:
            ContainerClient: A client scoped to the configured container.

        Raises:
            RuntimeError: If called for a non-Blob source.
        """
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient

        if self._container_url is None:
            raise RuntimeError("Azure Blob container URL is not configured")

        if has_sas_signature(self._container_url):
            with ContainerClient.from_container_url(container_url=self._container_url) as client:
                yield client
            return

        with DefaultAzureCredential() as credential:
            with ContainerClient.from_container_url(
                container_url=self._container_url,
                credential=credential,
            ) as client:
                yield client
