"""Local web application helpers for grax."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Return the local Flask application.

    Args:
        *args: Positional arguments forwarded to :func:`grax.web.app.create_app`.
        **kwargs: Keyword arguments forwarded to :func:`grax.web.app.create_app`.

    Returns:
        Configured Flask application.
    """
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
