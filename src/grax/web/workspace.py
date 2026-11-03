"""Choosing the database folder and browsing the local filesystem."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _safe_browse_path(raw_path: Any) -> Path:
    """Return a normalized filesystem path for the export dialog."""
    if raw_path in (None, ""):
        return Path.home().resolve()
    return Path(str(raw_path)).expanduser().resolve()


def _pick_native_directory() -> Path | None:
    """Open a platform folder dialog, returning ``None`` when the user cancels."""
    system = platform.system()
    if system == "Darwin":
        command = [
            "osascript",
            "-e",
            'POSIX path of (choose folder with prompt "Select GraxPy database folder" with invisibles)',
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
        if result.returncode:
            # AppleScript uses -128 for a user cancellation; other errors mean unavailable.
            if "-128" in result.stderr:
                return None
            raise RuntimeError(result.stderr.strip() or "macOS folder selection failed")
        selected = result.stdout.strip()
    elif system == "Windows":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError("PowerShell is not available")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select GraxPy database folder'; "
            "$dialog.ShowHiddenFiles = $true; "
            "if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath }"
        )
        result = subprocess.run(
            [powershell, "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Windows folder selection failed")
        selected = result.stdout.strip()
    elif system == "Linux":
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("no desktop session is available")
        python = shutil.which("python3")
        if python:
            gtk_script = (
                "import gi; gi.require_version('Gtk', '3.0'); "
                "from gi.repository import Gtk; "
                "dialog = Gtk.FileChooserDialog(title='Select GraxPy database folder', "
                "action=Gtk.FileChooserAction.SELECT_FOLDER); "
                "dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "
                "Gtk.STOCK_OPEN, Gtk.ResponseType.OK); "
                "dialog.set_show_hidden(True); "
                "response = dialog.run(); "
                "print(dialog.get_filename() or '') if response == Gtk.ResponseType.OK else None; "
                "dialog.destroy()"
            )
            gtk_result = subprocess.run(
                [python, "-c", gtk_script],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if gtk_result.returncode == 0:
                selected = gtk_result.stdout.strip()
                if not selected:
                    return None
                return Path(selected).expanduser().resolve()
        zenity = shutil.which("zenity")
        kdialog = shutil.which("kdialog")
        if zenity:
            result = subprocess.run(
                [
                    zenity,
                    "--file-selection",
                    "--directory",
                    "--title=Select GraxPy database folder",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        elif kdialog:
            result = subprocess.run(
                [
                    kdialog,
                    "--getexistingdirectory",
                    str(Path.home()),
                    "Select GraxPy database folder",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        else:
            raise RuntimeError("neither zenity nor kdialog is available")
        if result.returncode == 1:
            return None
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Linux folder selection failed")
        selected = result.stdout.strip()
    else:
        unsupported = system or "this platform"
        raise RuntimeError(f"native folder selection is not supported on {unsupported}")

    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def _validate_workspace_root(raw_path: Any) -> Path:
    """Return a writable workspace root path for the current session."""

    if raw_path in (None, ""):
        raise ValueError("Choose a workspace folder.")
    target_dir = Path(str(raw_path)).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    probe_path = target_dir / ".grax-web-write-test"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink()
    return target_dir


def _list_directory_entries(directory: Path) -> list[dict[str, Any]]:
    """Return browseable directory entries for the export dialog."""
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.iterdir(), key=lambda candidate: (not candidate.is_dir(), candidate.name.lower())):
        if path.name.startswith(".") or not path.is_dir():
            continue
        entries.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
            }
        )
    return entries


def _path_breadcrumbs(path: Path) -> list[dict[str, str]]:
    """Return breadcrumb metadata for one absolute filesystem path."""

    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    breadcrumbs = [{"name": str(anchor) if str(anchor) else resolved.parts[0], "path": str(anchor or resolved)}]
    current = anchor
    for part in resolved.parts[len(anchor.parts) :]:
        current = current / part
        breadcrumbs.append({"name": part, "path": str(current)})
    return breadcrumbs
