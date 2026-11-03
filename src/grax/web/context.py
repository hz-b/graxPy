"""Shared request-time helpers handed to every route group."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WebContext:
    """Accessors for the active workspace and its stores, bound to one app."""

    active_data_dir: Callable[..., Any]
    load_user_settings: Callable[..., Any]
    run_store: Callable[..., Any]
    store: Callable[..., Any]
