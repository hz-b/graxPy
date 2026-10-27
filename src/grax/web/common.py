"""Small shared helpers for the web app: IDs, versions, files and labels."""

from __future__ import annotations

import importlib.metadata
import ipaddress
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from grax.simulation.core import normalize_polarization

from .persistence import _slugify


def _split_variation_values(text: str) -> list[str]:
    """Split a comma-separated row of variation values."""

    return [value.strip() for value in str(text).split(",") if value.strip()]


def _relative_to(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string, or its name."""

    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def _unlink_relative(root: Path, relative: str | None) -> None:
    """Delete ``root / relative`` when it exists and stays inside ``root``."""

    if not relative:
        return
    candidate = (root / relative).resolve()
    if root.resolve() in candidate.parents and candidate.exists():
        candidate.unlink()


def _normalized_polarization(value: Any) -> str:
    """Return a canonical polarization code for Web UI run inputs.

    Delegates to :func:`grax.normalize_polarization`, so the form accepts the
    same ``TE``/``TM`` aliases the Python API does and stores the canonical
    ``"s"``/``"p"``.
    """

    return normalize_polarization(value or "p")


def _normalized_solver(value: Any) -> str:
    """Return a validated electromagnetic solver name for Web UI run inputs."""

    solver = str(value or "rcwa").strip().lower()
    if solver not in {"rcwa", "neviere"}:
        raise ValueError("solver must be 'rcwa' or 'neviere'.")
    return solver


def _write_text_atomically(path: Path, text: str) -> None:
    """Write diagnostic text through a temporary file and replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _site_metadata() -> dict[str, str]:
    """Return shared attribution metadata for the web UI."""

    return {
        "version": _grax_version(),
        "organization": "Helmholtz-Zentrum Berlin",
        "contact_name": "Simone Vadilonga",
        "contact_email": "simone.vadilonga@helmholtz-berlin.de",
        "github_url": "https://github.com/hz-b/graxPy",
        "docs_url": "https://graxpy.readthedocs.io/en/latest/",
    }


def _grax_version() -> str:
    """Return the project version, preferring pyproject.toml in a source checkout."""

    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            project = tomllib.load(pyproject_file).get("project", {})
        version = str(project.get("version", "")).strip()
        if version:
            return version
    except (OSError, tomllib.TOMLDecodeError):
        pass

    try:
        return importlib.metadata.version("graxpy")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _file_token(path: Path) -> str:
    """Return a cheap cache-busting token for one file path."""

    if not path.exists():
        return ""
    return str(path.stat().st_mtime_ns)


def _polarization_display_label(value: Any) -> str:
    """Return the conventional S/TE or P/TM label for a stored code."""

    code = str(value).strip().lower()
    if code in {"s", "te"}:
        return "S (TE)"
    if code in {"p", "tm"}:
        return "P (TM)"
    return str(value)


def _plot_id() -> str:
    """Return a filesystem-safe plot identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _run_id(workflow: str, grating_id: str) -> str:
    """Return a filesystem-safe run ID."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{grating_id}-{workflow}".replace("_", "-")


PATH_ID_KEYS = ("run_id", "plot_id", "study_id", "mirror_id", "scan_id", "grating_id")


def _is_safe_id(value: Any) -> bool:
    """Return whether an ID is a plain slug that cannot leave its store folder."""
    return isinstance(value, str) and bool(value) and _slugify(value) == value


def _is_cross_site_request(*, origin: str | None, fetch_site: str | None, host_url: str) -> bool:
    """Return whether browser headers show the request came from another site's page."""
    if (fetch_site or "").lower() == "cross-site":
        return True
    if not origin:
        return False
    if origin == "null":
        return (fetch_site or "").lower() not in {"same-origin", "none"}
    origin_parts = urlsplit(origin)
    host_parts = urlsplit(host_url)
    return (origin_parts.scheme, origin_parts.netloc) != (host_parts.scheme, host_parts.netloc)


def _is_loopback_address(raw_address: str | None) -> bool:
    """Return whether a request came from the local machine."""
    if not raw_address:
        return False
    try:
        address = ipaddress.ip_address(raw_address)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback
