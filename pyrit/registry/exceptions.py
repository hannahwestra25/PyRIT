# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exceptions raised by registry validation."""


class RegistryValidationError(ValueError):
    """Raised when registry-supplied arguments cannot be safely resolved or configured."""
