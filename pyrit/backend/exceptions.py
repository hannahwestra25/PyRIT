# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Backend exceptions with client-safe messages."""


class ClientRequestError(ValueError):
    """A request error whose message is safe to return to API clients."""
