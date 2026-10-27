"""Side-by-side parity check between two versions of the grax web app.

Starts a *baseline* and a *candidate* server, each from its own ``src`` tree and
each on its own copy of a real ``.grax-web`` database, sends both the same
requests, and reports every response or run artifact that differs.

The source database is only read; every write happens in the copies under
``--work``. Typical use (the baseline is a git worktree of the reference commit):

    git worktree add /tmp/grax-baseline develop
    .venv/bin/python tests/characterization/web_parity.py \
        --baseline-src /tmp/grax-baseline/src --candidate-src src \
        --database .grax-web --work /tmp/grax-parity --tier quick

Tiers:
    read   pages, JSON endpoints, exports and plot previews of the saved data
    quick  read + replay every saved run, mirror run and multilayer study with
           each scan axis thinned to a few points, plus run lifecycle and plot edits
    full   quick + replay a selection of small runs at full size and compare
           them with the results stored in the original database
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"completed", "failed", "aborted", "error", "stale"}
VOLATILE_JSON_KEYS = {
    "created_at", "updated_at", "elapsed_seconds", "eta_seconds", "plot_token",
    "resolved_workers", "started_at", "finished_at", "pid", "memory", "used_bytes",
    "available_bytes", "total_bytes", "duration_seconds", "timestamp", "saved_at",
}
SERVER_SCRIPT = """\
import sys

if __name__ == "__main__":
    from werkzeug.serving import make_server

    import grax
    from grax.web.app import create_app

    print("GRAX_SOURCE", grax.__file__, flush=True)
    app = create_app(data_dir=sys.argv[1])
    make_server("127.0.0.1", int(sys.argv[2]), app, threaded=True).serve_forever()
"""


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D102
        return None


@dataclass
class Reply:
    """One HTTP response."""

    status: int
    content_type: str
    body: bytes
    location: str = ""


@dataclass
class Side:
    """One running server (baseline or candidate) and its private data copy."""

    name: str
    src: Path
    root: Path
    port: int
    process: subprocess.Popen | None = None
    opener: Any = field(default_factory=lambda: urllib.request.build_opener(_NoRedirect))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(self, method: str, path: str, data: Any = None, headers: dict | None = None) -> Reply:
        body = None
        if data is not None:
            body = urllib.parse.urlencode(_form_items(data), doseq=True).encode()
        request = urllib.request.Request(self.base_url + path, data=body, method=method, headers=headers or {})
        try:
            with self.opener.open(request, timeout=600) as response:
                return Reply(response.status, response.headers.get("Content-Type", ""), response.read(),
                             response.headers.get("Location", ""))
        except urllib.error.HTTPError as error:
            return Reply(error.code, error.headers.get("Content-Type", ""), error.read(),
                         error.headers.get("Location", ""))
        except (OSError, ValueError) as error:
            return Reply(-1, "", f"{type(error).__name__}: {error}".encode())

    def get(self, path: str) -> Reply:
        return self.request("GET", path)

    def post(self, path: str, data: Any) -> Reply:
        return self.request("POST", path, data)


def _form_items(data: Any) -> list[tuple[str, str]]:
    items = data.items() if isinstance(data, dict) else data
    pairs: list[tuple[str, str]] = []
    for key, value in items:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, (dict, list)):
                item = json.dumps(item)
            pairs.append((key, "" if item is None else str(item)))
    return pairs


def _redirect_path(location: str) -> str:
    parts = urllib.parse.urlsplit(location)
    return parts.path + (f"?{parts.query}" if parts.query else "")


# ---------------------------------------------------------------------------
# Normalization and comparison
# ---------------------------------------------------------------------------


@dataclass
class Report:
    """Collected parity results."""

    equal: int = 0
    differences: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ok(self) -> None:
        with self.lock:
            self.equal += 1

    def diff(self, label: str, detail: str) -> None:
        with self.lock:
            self.differences.append((label, detail))
        print(f"  DIFF {label}", flush=True)

    def note(self, text: str) -> None:
        with self.lock:
            self.notes.append(text)
        print(f"  NOTE {text}", flush=True)


class Normalizer:
    """Mask values that legitimately differ between the two servers."""

    def __init__(self, side: Side) -> None:
        self.side = side
        self.aliases: dict[str, str] = {}

    def alias(self, value: str, placeholder: str) -> None:
        if value:
            self.aliases[value] = placeholder

    def text(self, text: str) -> str:
        for value, placeholder in sorted(self.aliases.items(), key=lambda item: -len(item[0])):
            text = text.replace(value, placeholder)
            text = text.replace(urllib.parse.quote(value), placeholder)
        for path in {str(self.side.data), str(self.side.data.resolve()), str(self.side.root.resolve())}:
            text = text.replace(path, "<DATA>")
        text = re.sub(r"127\.0\.0\.1:\d+", "<HOST>", text)
        # Which files carry the page scripts is layout, not behavior (web.js was split into static/js/).
        text = re.sub(r"[ \t]*<script src=\"/static/[^\"]+\.js[^\"]*\" defer></script>\n", "", text)
        text = re.sub(r"\n[ \t]*(?=\n)", "", text)  # blank lines left by template loops
        text = re.sub(r"([?&](?:v|t|token)=)[0-9A-Za-z.\-_]+", r"\1<V>", text)
        text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?", "<TIME>", text)
        text = re.sub(r"\b\d{8}-\d{6}(?:-\d{6})?\b", "<STAMP>", text)
        text = re.sub(r"\b[0-9a-f]{32}\b", "<UUID>", text)
        text = re.sub(r"\b[0-9a-f]{12}\b", "<HEX12>", text)
        return text

    def json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ("<VOLATILE>" if key in VOLATILE_JSON_KEYS else self.json(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self.json(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def _numeric(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def compare_tables(label: str, left: str, right: str, report: Report, *, rtol: float = 0.0) -> None:
    """Compare two CSV/TSV texts as unordered row sets with optional tolerance."""
    def rows(text: str) -> list[list[str]]:
        first_line = text.splitlines()[0] if text else ""
        return list(csv.reader(io.StringIO(text), dialect="excel-tab" if "\t" in first_line else "excel"))

    left_rows, right_rows = rows(left), rows(right)
    if left_rows[:1] != right_rows[:1]:
        report.diff(label, f"header differs: {left_rows[:1]} vs {right_rows[:1]}")
        return
    if len(left_rows) != len(right_rows):
        report.diff(label, f"row count differs: {len(left_rows)} vs {len(right_rows)}")
        return
    left_sorted, right_sorted = sorted(left_rows[1:]), sorted(right_rows[1:])
    for left_row, right_row in zip(left_sorted, right_sorted):
        if left_row == right_row:
            continue
        if len(left_row) != len(right_row):
            report.diff(label, f"row width differs: {left_row} vs {right_row}")
            return
        for a, b in zip(left_row, right_row):
            if a == b:
                continue
            fa, fb = _numeric(a), _numeric(b)
            if fa is None or fb is None or not (
                (math.isnan(fa) and math.isnan(fb)) or math.isclose(fa, fb, rel_tol=rtol, abs_tol=rtol * 1e-3)
            ):
                report.diff(label, f"value differs: {left_row} vs {right_row}")
                return
    report.ok()


def compare_bodies(label: str, kind: str, left: bytes, right: bytes, norm_left: Normalizer,
                   norm_right: Normalizer, report: Report) -> None:
    """Compare two payloads of the given kind after normalization."""
    if kind == "json":
        try:
            a = norm_left.json(json.loads(left))
            b = norm_right.json(json.loads(right))
        except ValueError:
            kind = "text"
        else:
            if a == b:
                report.ok()
            elif _sorted_lists(a) == _sorted_lists(b):
                report.note(f"{label}: same content, different list order")
            else:
                _text_diff(label, json.dumps(a, indent=1, sort_keys=True),
                           json.dumps(b, indent=1, sort_keys=True), report)
            return
    if kind == "zip":
        compare_zips(label, left, right, norm_left, norm_right, report)
        return
    if kind == "binary":
        if left == right:
            report.ok()
        else:
            report.note(f"{label}: binary payload differs ({len(left)} vs {len(right)} bytes)")
        return
    a = norm_left.text(left.decode("utf-8", "replace"))
    b = norm_right.text(right.decode("utf-8", "replace"))
    if kind == "csv":
        compare_tables(label, a, b, report)
    elif a == b:
        report.ok()
    elif sorted(a.splitlines()) == sorted(b.splitlines()):
        report.note(f"{label}: same lines, different order")
    else:
        _text_diff(label, a, b, report)


def _sorted_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_lists(item) for key, item in value.items()}
    if isinstance(value, list):
        return sorted((_sorted_lists(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return value


def _text_diff(label: str, a: str, b: str, report: Report) -> None:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "baseline", "candidate", n=1, lineterm=""))
    report.diff(label, "\n".join(lines[:60]))


def _kind_for(name: str, content_type: str = "") -> str:
    lowered = name.lower().split("?")[0]
    if lowered.endswith((".csv", ".dat", ".tsv", "/export/csv", "/export/dat")) or "text/csv" in content_type:
        return "csv"
    if lowered.endswith(".json") or "json" in content_type:
        return "json"
    if lowered.endswith(".zip") or "zip" in content_type:
        return "zip"
    if lowered.endswith((".png", ".jpg", ".pdf", ".npz", ".npy")) or content_type.startswith("image/"):
        return "binary"
    return "text"


def compare_zips(label: str, left: bytes, right: bytes, norm_left: Normalizer, norm_right: Normalizer,
                 report: Report) -> None:
    with zipfile.ZipFile(io.BytesIO(left)) as a, zipfile.ZipFile(io.BytesIO(right)) as b:
        names_a = {norm_left.text(name): name for name in a.namelist()}
        names_b = {norm_right.text(name): name for name in b.namelist()}
        if set(names_a) != set(names_b):
            report.diff(label, f"zip members differ: {sorted(set(names_a) ^ set(names_b))}")
            return
        for key in sorted(names_a):
            compare_bodies(f"{label}::{key}", _kind_for(key), a.read(names_a[key]), b.read(names_b[key]),
                           norm_left, norm_right, report)


def compare_replies(label: str, left: Reply, right: Reply, norm_left: Normalizer, norm_right: Normalizer,
                    report: Report) -> None:
    if left.status != right.status:
        report.diff(label, f"status {left.status} vs {right.status}")
        return
    if norm_left.text(left.location) != norm_right.text(right.location):
        report.diff(label, f"redirect {left.location} vs {right.location}")
        return
    compare_bodies(label, _kind_for(label, left.content_type), left.body, right.body, norm_left, norm_right, report)


def compare_dirs(label: str, left: Path, right: Path, norm_left: Normalizer, norm_right: Normalizer,
                 report: Report, *, skip: tuple[str, ...] = ("checkpoints",)) -> None:
    """Compare two run/study directories file by file."""
    def files(root: Path, norm: Normalizer) -> dict[str, Path]:
        return {
            norm.text(str(path.relative_to(root))): path
            for path in root.rglob("*")
            if path.is_file() and not any(part in skip for part in path.relative_to(root).parts)
        }

    files_a, files_b = files(left, norm_left), files(right, norm_right)
    if set(files_a) != set(files_b):
        report.diff(f"{label} files", f"only baseline: {sorted(set(files_a) - set(files_b))}; "
                                      f"only candidate: {sorted(set(files_b) - set(files_a))}")
    for key in sorted(set(files_a) & set(files_b)):
        compare_bodies(f"{label}/{key}", _kind_for(key), files_a[key].read_bytes(), files_b[key].read_bytes(),
                       norm_left, norm_right, report)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


LINK_PATTERN = re.compile(r'(?:href|src)="(/[^"#\s<>()]*)"')
SCRIPT_PATTERN = re.compile(r"<script\b[^>]*>.*?</script>", re.S)


class FormParser(HTMLParser):
    """Collect the successful controls of every <form>, as a browser would submit them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[dict[str, Any]] = []
        self._select: tuple[str, list[str], str | None] | None = None
        self._textarea: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "form":
            self.forms.append({"action": values.get("action", ""), "method": values.get("method", "get").lower(),
                               "fields": []})
            return
        if not self.forms or "disabled" in values:
            return
        fields = self.forms[-1]["fields"]
        name = values.get("name")
        if tag == "input" and name:
            kind = values.get("type", "text").lower()
            if kind in {"submit", "button", "file", "image", "reset"}:
                return
            if kind in {"checkbox", "radio"} and "checked" not in values:
                return
            fields.append((name, values.get("value", "on" if kind in {"checkbox", "radio"} else "")))
        elif tag == "select" and name:
            self._select = (name, [], None)
        elif tag == "option" and self._select is not None:
            name, options, chosen = self._select
            options.append(values.get("value", ""))
            if "selected" in values:
                chosen = values.get("value", "")
            self._select = (name, options, chosen)
        elif tag == "textarea" and name:
            self._textarea, self._text = name, []

    def handle_data(self, data: str) -> None:
        if self._textarea is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._select is not None and self.forms:
            name, options, chosen = self._select
            if chosen is not None or options:
                self.forms[-1]["fields"].append((name, chosen if chosen is not None else options[0]))
            self._select = None
        elif tag == "textarea" and self._textarea is not None and self.forms:
            self.forms[-1]["fields"].append((self._textarea, "".join(self._text).lstrip("\n")))
            self._textarea = None


def page_forms(html: str) -> list[dict[str, Any]]:
    parser = FormParser()
    parser.feed(html)
    return parser.forms


# ---------------------------------------------------------------------------
# Parity driver
# ---------------------------------------------------------------------------


class Parity:
    """Drive both servers through the same scenario."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = Path(args.work).resolve()
        self.database = Path(args.database).resolve()
        self.baseline = Side("baseline", Path(args.baseline_src).resolve(), self.work / "baseline", args.port)
        self.candidate = Side("candidate", Path(args.candidate_src).resolve(), self.work / "candidate", args.port + 1)
        self.norms = {side.name: Normalizer(side) for side in (self.baseline, self.candidate)}
        self.report = Report()
        self.points = args.points

    # -- setup -------------------------------------------------------------

    def prepare(self) -> None:
        if self.work.exists():
            shutil.rmtree(self.work)
        for side in (self.baseline, self.candidate):
            side.root.mkdir(parents=True)
            # Snapshot the code so edits made while the comparison runs cannot leak into it.
            shutil.copytree(side.src, side.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            # The site reads its version from the pyproject.toml next to src/.
            if (side.src.parent / "pyproject.toml").is_file():
                shutil.copy2(side.src.parent / "pyproject.toml", side.root / "pyproject.toml")
            side.src = side.root / "src"
            shutil.copytree(self.database, side.data, symlinks=True)
            (side.root / "serve.py").write_text(SERVER_SCRIPT, encoding="utf-8")

    def start(self) -> None:
        for side in (self.baseline, self.candidate):
            env = {**os.environ, "PYTHONPATH": str(side.src), "MPLBACKEND": "Agg"}
            log = (side.root / "server.log").open("w", encoding="utf-8")
            side.process = subprocess.Popen(
                [sys.executable, str(side.root / "serve.py"), str(side.data), str(side.port)],
                cwd=side.root, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        for side in (self.baseline, self.candidate):
            deadline = time.time() + 120
            while True:
                try:
                    if side.get("/").status == 200:
                        break
                except OSError:
                    pass
                if time.time() > deadline or side.process.poll() is not None:
                    raise RuntimeError(f"{side.name} server did not start; see {side.root / 'server.log'}")
                time.sleep(0.5)
            source = (side.root / "server.log").read_text(encoding="utf-8").split("GRAX_SOURCE", 1)[-1].split()[0]
            print(f"{side.name}: {side.base_url} serving grax from {source}", flush=True)
            if not Path(source).resolve().is_relative_to(side.src):
                raise RuntimeError(f"{side.name} imports grax from {source}, not from {side.src}")

    def stop(self) -> None:
        for side in (self.baseline, self.candidate):
            if side.process and side.process.poll() is None:
                side.process.terminate()
                try:
                    side.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    side.process.kill()

    # -- generic request helpers --------------------------------------------

    def both(self, method: str, path: str, data: Any = None, *, label: str | None = None,
             path_candidate: str | None = None, compare: bool = True) -> tuple[Reply, Reply]:
        with ThreadPoolExecutor(2) as pool:
            left = pool.submit(self.baseline.request, method, path, data)
            right = pool.submit(self.candidate.request, method, path_candidate or path, data)
            a, b = left.result(), right.result()
        if compare:
            compare_replies(label or f"{method} {path}", a, b, self.norms["baseline"], self.norms["candidate"],
                            self.report)
        return a, b

    def wait_for(self, side: Side, status_path: str, *, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        payload: dict[str, Any] = {}
        while time.time() < deadline:
            reply = side.get(status_path)
            if reply.status == 200:
                payload = json.loads(reply.body)
                state = str(payload.get("state") or payload.get("status") or "")
                if state in TERMINAL_STATES:
                    return payload
            time.sleep(0.5)
        raise TimeoutError(f"{side.name}: {status_path} did not finish; last payload {payload}")

    def wait_both(self, status_path: str, status_path_candidate: str | None = None, *,
                  timeout: float = 1800) -> tuple[dict, dict]:
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(self.wait_for, self.baseline, status_path, timeout=timeout)
            b = pool.submit(self.wait_for, self.candidate, status_path_candidate or status_path, timeout=timeout)
            return a.result(), b.result()

    # -- phase: read-only ------------------------------------------------------

    def crawl(self) -> None:
        print("== read-only crawl", flush=True)
        seen: set[str] = set()
        queue = ["/"]
        while queue:
            path = queue.pop(0)
            if path in seen:
                continue
            seen.add(path)
            a, b = self.both("GET", path)
            if a.status != 200 or "text/html" not in a.content_type:
                continue
            for link in LINK_PATTERN.findall(SCRIPT_PATTERN.sub("", a.body.decode("utf-8", "replace"))):
                target = urllib.parse.urljoin(path, link.replace("&amp;", "&"))
                parts = urllib.parse.urlsplit(target)
                if parts.scheme or parts.netloc or parts.path.startswith(("/static/", "/_data/previews/live/")):
                    continue  # live previews get a fresh random name per request; compared in previews()
                if "/abort" in parts.path:
                    continue
                queue.append(parts.path + (f"?{parts.query}" if parts.query else ""))
        print(f"   crawled {len(seen)} URLs", flush=True)

    def endpoints(self) -> None:
        print("== JSON endpoints and exports", flush=True)
        data = self.baseline.data
        for run_dir in sorted((data / "runs").iterdir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            run_id = run_dir.name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("mirror_id"):
                paths = [f"/plane-mirror-runs/{run_id}/status", f"/plane-mirror-runs/{run_id}/stack-preview.png"]
            else:
                paths = [f"/runs/{run_id}/status", f"/runs/{run_id}/export-all", f"/runs/{run_id}/energy-tables"]
                paths += [f"/runs/{run_id}/export/{fmt}" for fmt in ("csv", "dat", "yaml")]
                paths += [f"/runs/{run_id}/scan-series?order={order}" for order in (0, 1, 2, -1)]
                paths.append(f"/runs/{run_id}/abort")
            for path in paths:
                self.both("GET", path)
        for plot_dir in sorted((data / "plots").glob("*/")):
            for kind in ("data", "script"):
                self.both("GET", f"/plots/{plot_dir.name}/export/{kind}")
            self.both("GET", f"/plots/{plot_dir.name}/delete")
        for study_dir in sorted((data / "multilayer_designs").glob("*/")):
            study = study_dir.name
            for path in ("script", "survey-options", "energy-scan-points"):
                self.both("GET", f"/multilayer-design/{study}/{path}")
            for stage in ("survey", "energy_scan"):
                self.both("GET", f"/multilayer-design/{study}/stages/{stage}/status")
        for mirror in sorted((data / "saved_plane_mirrors").glob("*.json")):
            spec = json.loads(mirror.read_text(encoding="utf-8"))
            self.both("GET", f"/plane-mirrors/{mirror.stem}/stack-preview.png")
            for scan in spec.get("scans", []):
                self.both("GET", f"/plane-mirrors/{mirror.stem}/scans/{scan['id']}/preview.png")
        self.both("GET", "/plots/export")
        for path in ("/system/resource-status", "/system/memory"):  # machine load differs: status only
            a, b = self.both("GET", path, compare=False)
            if a.status != b.status:
                self.report.diff(f"GET {path}", f"status {a.status} vs {b.status}: {b.body[:300]!r}")
            else:
                self.report.ok()

    def previews(self) -> None:
        """Submit each page's own forms to the side-effect-free preview endpoints."""
        print("== preview endpoints from page forms", flush=True)
        data = self.baseline.data
        pages = ["/gratings/new", "/plane-mirror", "/plots/new"]
        pages += [f"/gratings/{path.stem}/edit" for path in sorted((data / "saved_gratings").glob("*.json"))]
        pages += [f"/plots/{path.name}/edit" for path in sorted((data / "plots").glob("*/"))]
        pages += [f"/plane-mirrors/{path.stem}" for path in sorted((data / "saved_plane_mirrors").glob("*.json"))]
        previews = {
            "/gratings": ["/_preview/grating"],
            "/plots": ["/_preview/plot"],
            "/plane-mirrors": ["/_preview/plane-mirror-stack", "/_compute/plane-mirror", "/plane-mirror/csv"],
        }
        for page in pages:
            reply = self.baseline.get(page)
            if reply.status != 200:
                continue
            for form in page_forms(reply.body.decode("utf-8", "replace")):
                if form["method"] != "post" or not form["fields"]:
                    continue
                action = urllib.parse.urlsplit(form["action"]).path
                for prefix, targets in previews.items():
                    if action.startswith(prefix) or (prefix == "/plots" and action.startswith("/runs/compare")):
                        for target in targets:
                            label = f"POST {target} (form of {page})"
                            a, b = self.both("POST", target, form["fields"], label=label)
                            self.compare_preview_images(label, a, b)

    def compare_preview_images(self, label: str, a: Reply, b: Reply) -> None:
        if a.status != 200 or "json" not in a.content_type or b.status != 200:
            return
        payload_a, payload_b = json.loads(a.body), json.loads(b.body)
        for key in ("preview_url", "roughness_preview_url", "stack_preview_url"):
            url_a, url_b = payload_a.get(key), payload_b.get(key)
            if isinstance(url_a, str) and isinstance(url_b, str) and url_a.startswith("/"):
                image_a, image_b = self.baseline.get(url_a), self.candidate.get(url_b)
                compare_replies(f"{label} {key}", image_a, image_b, self.norms["baseline"],
                                self.norms["candidate"], self.report)

    # -- phase: replay ------------------------------------------------------------

    def thin(self, value: Any) -> Any:
        """Reduce every scan axis to at most ``self.points`` entries."""
        limit = self.points
        if isinstance(value, dict):
            thinned = {}
            for key, item in value.items():
                if key in {"values", "rows"} and isinstance(item, list):
                    thinned[key] = [self.thin(entry) for entry in item[:limit]]
                elif key in {"count", "points"} and isinstance(item, (int, float)) and item > limit:
                    thinned[key] = limit
                else:
                    thinned[key] = self.thin(item)
            return thinned
        if isinstance(value, list):
            return [self.thin(item) for item in value]
        return value

    def thin_form(self, form: dict[str, Any]) -> dict[str, Any]:
        thinned = dict(form)
        for key, value in form.items():
            if key.endswith("_points") or key == "points":
                number = _numeric(str(value)) if not isinstance(value, list) else None
                if number is not None and number > self.points:
                    thinned[key] = str(self.points)
        return thinned

    @staticmethod
    def decode_config(value: Any) -> Any:
        if isinstance(value, str) and value.strip().startswith(("{", "[")):
            try:
                return json.loads(value)
            except ValueError:
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
        return value

    def grating_run_form(self, manifest: dict[str, Any], *, thin: bool) -> dict[str, Any]:
        form = {key: self.decode_config(value) for key, value in (manifest.get("run_input") or {}).items()}
        if "custom_grid" in manifest and isinstance(manifest["custom_grid"], dict):
            form["custom_grid_config"] = manifest["custom_grid"]
        # One worker keeps result order deterministic; parallel runs append in completion order.
        form["max_workers_mode"], form["max_workers"] = "manual", "1"
        if thin:
            form = self.thin_form(form)
            for key in ("custom_grid_config", "parameter_sweep_config"):
                if isinstance(form.get(key), dict):
                    form[key] = self.thin(form[key])
        return form

    def run_signature(self, manifest: dict[str, Any], form: dict[str, Any]) -> str:
        return hashlib.sha1(json.dumps([manifest.get("grating_spec"), form], sort_keys=True,
                                       default=str).encode()).hexdigest()

    def install(self, relative: str, payload: dict[str, Any]) -> None:
        for side in (self.baseline, self.candidate):
            path = side.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def replay_grating_runs(self, *, full: bool) -> None:
        print(f"== replay grating runs ({'full size' if full else f'thinned to {self.points} points'})", flush=True)
        seen: set[str] = set()
        for index, manifest_path in enumerate(sorted((self.database / "runs").glob("*/manifest.json"))):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("mirror_id") or not manifest.get("grating_spec") or not manifest.get("run_input"):
                continue
            if full and int(manifest.get("total_points") or 0) > self.args.full_max_points:
                continue
            if full and manifest.get("status") != "completed":
                continue
            form = self.grating_run_form(manifest, thin=not full)
            signature = self.run_signature(manifest, form)
            if signature in seen:
                continue
            seen.add(signature)
            grating_id = f"replay-{'full' if full else 'quick'}-{index:03d}"
            spec = {**manifest["grating_spec"], "id": grating_id, "name": f"Replay {index:03d}"}
            self.install(f"saved_gratings/{grating_id}.json", spec)
            label = f"{manifest.get('workflow')} {manifest_path.parent.name}"
            print(f" - {label}", flush=True)
            a, b = self.both("POST", f"/gratings/{grating_id}/runs", form, label=f"start {label}", compare=False)
            if a.status != b.status:
                self.report.diff(f"start {label}", f"status {a.status} vs {b.status}: "
                                                   f"{a.body[:300]!r} / {b.body[:300]!r}")
                continue
            if a.status != 302:
                compare_replies(f"start {label}", a, b, self.norms["baseline"], self.norms["candidate"], self.report)
                self.report.note(f"{label}: both servers rejected the replay with {a.status}")
                continue
            run_a = _redirect_path(a.location).rstrip("/").split("/")[-1]
            run_b = _redirect_path(b.location).rstrip("/").split("/")[-1]
            self.norms["baseline"].alias(run_a, "<RUN>")
            self.norms["candidate"].alias(run_b, "<RUN>")
            try:
                status_a, status_b = self.wait_both(f"/runs/{run_a}/status", f"/runs/{run_b}/status")
            except TimeoutError as error:
                self.report.diff(label, str(error))
                continue
            self.compare_json(f"{label} final status", status_a, status_b)
            self.compare_run(label, run_a, run_b)
            if full:
                self.compare_with_original(label, manifest_path.parent, self.candidate.data / "runs" / run_b)
            self.norms["baseline"].aliases.pop(run_a, None)
            self.norms["candidate"].aliases.pop(run_b, None)

    def compare_json(self, label: str, a: Any, b: Any) -> None:
        compare_bodies(label, "json", json.dumps(a).encode(), json.dumps(b).encode(), self.norms["baseline"],
                       self.norms["candidate"], self.report)

    def compare_run(self, label: str, run_a: str, run_b: str) -> None:
        for suffix in ("", "/export-all", "/energy-tables", "/export/csv", "/export/dat", "/export/yaml"):
            self.both("GET", f"/runs/{run_a}{suffix}", label=f"{label} GET run{suffix}",
                      path_candidate=f"/runs/{run_b}{suffix}")
        compare_dirs(f"{label} run dir", self.baseline.data / "runs" / run_a, self.candidate.data / "runs" / run_b,
                     self.norms["baseline"], self.norms["candidate"], self.report)

    def compare_with_original(self, label: str, original: Path, replay: Path) -> None:
        for name in ("all_orders.csv", "summary.csv"):
            if (original / name).is_file() and (replay / name).is_file():
                compare_tables(f"{label} vs stored {name}", (original / name).read_text(encoding="utf-8"),
                               (replay / name).read_text(encoding="utf-8"), self.report, rtol=self.args.rtol)

    def replay_plane_mirror_runs(self) -> None:
        print("== replay plane-mirror runs (thinned)", flush=True)
        for index, manifest_path in enumerate(sorted((self.database / "runs").glob("*/manifest.json"))):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("mirror_id") or not manifest.get("cases"):
                continue
            mirror_id = f"replay-mirror-{index:03d}"
            snapshot = manifest.get("mirror_snapshot") or {}
            self.install(f"saved_plane_mirrors/{mirror_id}.json", {
                "id": mirror_id, "name": f"Replay mirror {index:03d}", "label": "", "schema_version": 4,
                "stack_form": snapshot.get("stack_form", {}), "scans": [],
                "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00",
            })
            form: dict[str, Any] = self.thin_form(dict(manifest["cases"][0]["scan"]))
            axes = manifest.get("variation_axes") or []
            form["axis_parameter"] = [axis["parameter"] for axis in axes]
            form["axis_start"] = [str(axis["start"]) for axis in axes]
            form["axis_stop"] = [str(axis["stop"]) for axis in axes]
            form["axis_points"] = [str(min(int(float(axis["points"])), self.points)) for axis in axes]
            lefts, left_values, rights, right_values = [], [], [], []
            for row in manifest.get("variation_coupled") or []:
                (left, lv), (right, rv) = list(row.items())[:2]
                lefts.append(left)
                rights.append(right)
                left_values.append(", ".join(str(item) for item in lv[: self.points]))
                right_values.append(", ".join(str(item) for item in rv[: self.points]))
            form.update(coupled_left_parameter=lefts, coupled_left_values=left_values,
                        coupled_right_parameter=rights, coupled_right_values=right_values,
                        run_name=f"Replay {index:03d}")
            label = f"plane_mirror {manifest_path.parent.name}"
            print(f" - {label}", flush=True)
            a, b = self.both("POST", f"/plane-mirrors/{mirror_id}/runs", form, label=f"start {label}", compare=False)
            if a.status != b.status or a.status != 302:
                compare_replies(f"start {label}", a, b, self.norms["baseline"], self.norms["candidate"], self.report)
                continue
            run_a = _redirect_path(a.location).rstrip("/").split("/")[-1]
            run_b = _redirect_path(b.location).rstrip("/").split("/")[-1]
            self.norms["baseline"].alias(run_a, "<RUN>")
            self.norms["candidate"].alias(run_b, "<RUN>")
            status_a, status_b = self.wait_both(f"/plane-mirror-runs/{run_a}/status",
                                                f"/plane-mirror-runs/{run_b}/status")
            self.compare_json(f"{label} final status", status_a, status_b)
            self.both("GET", f"/plane-mirror-runs/{run_a}", label=f"{label} detail",
                      path_candidate=f"/plane-mirror-runs/{run_b}")
            compare_dirs(f"{label} run dir", self.baseline.data / "runs" / run_a,
                         self.candidate.data / "runs" / run_b, self.norms["baseline"], self.norms["candidate"],
                         self.report)

    def replay_multilayer(self) -> None:
        print("== replay multilayer studies (thinned)", flush=True)
        for study_path in sorted((self.database / "multilayer_designs").glob("*/study.json")):
            study = json.loads(study_path.read_text(encoding="utf-8"))
            study_id = study_path.parent.name
            config = study["config"]
            config["max_workers"] = 1
            for key in ("d_points", "blaze_points", "energy_scan_points"):
                config[key] = min(int(config.get(key) or 2), 2 if key != "energy_scan_points" else self.points)
            for settings_key in ("survey_scan_settings", "energy_scan_settings"):
                settings = config.get(settings_key) or {}
                for key, value in list(settings.items()):
                    if key.endswith("_points") and isinstance(value, int):
                        settings[key] = min(value, 11)
            for side in (self.baseline, self.candidate):
                target = side.data / "multilayer_designs" / study_id / "study.json"
                manifest = json.loads(target.read_text(encoding="utf-8"))
                manifest["config"] = config
                target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            label = f"multilayer {study_id}"
            print(f" - {label}", flush=True)
            for stage in ("survey", "energy_scan"):
                self.both("POST", f"/multilayer-design/{study_id}/stages/{stage}/reset", {},
                          label=f"{label} reset {stage}")
            self.both("POST", f"/multilayer-design/{study_id}/stages/survey/run", {}, label=f"{label} run survey")
            self.wait_both(f"/multilayer-design/{study_id}/stages/survey/status", timeout=3600)
            time.sleep(2)  # auto energy scan may queue right after the survey
            status = json.loads(self.baseline.get(f"/multilayer-design/{study_id}/stages/energy_scan/status").body)
            if str(status.get("state")) in {"queued", "running"}:
                self.wait_both(f"/multilayer-design/{study_id}/stages/energy_scan/status", timeout=3600)
            elif str(status.get("state")) != "completed":
                self.both("POST", f"/multilayer-design/{study_id}/stages/energy_scan/run", {"mode": "best"},
                          label=f"{label} run energy scan")
                self.wait_both(f"/multilayer-design/{study_id}/stages/energy_scan/status", timeout=3600)
            for stage in ("survey", "energy_scan"):
                self.both("GET", f"/multilayer-design/{study_id}/stages/{stage}/status", label=f"{label} {stage}")
            for path in ("", "/script", "/survey-options", "/energy-scan-points"):
                self.both("GET", f"/multilayer-design/{study_id}{path}", label=f"{label} GET{path}")
            compare_dirs(label, self.baseline.data / "multilayer_designs" / study_id,
                         self.candidate.data / "multilayer_designs" / study_id, self.norms["baseline"],
                         self.norms["candidate"], self.report)

    def lifecycle(self) -> None:
        """Start, abort (save), resume, restart and abort (delete) one run on both servers."""
        print("== run lifecycle", flush=True)
        grating = sorted((self.database / "saved_gratings").glob("*.json"))[0].stem
        form = {
            "workflow": "energy_sweep", "energy_start_ev": "100", "energy_stop_ev": "400", "energy_points": "40",
            "grazing_angle_deg": "1.5", "diffraction_order": "1", "fourier_orders": "5", "polarization": "s",
            "solver": "rcwa", "roughness_kind": "none", "run_x_resolution_nm": "2", "run_z_resolution_nm": "2",
            "max_workers_mode": "manual", "max_workers": "1", "comment": "parity lifecycle",
        }
        a, b = self.both("POST", f"/gratings/{grating}/runs", form, compare=False, label="lifecycle start")
        if a.status != 302 or b.status != 302:
            self.report.diff("lifecycle start", f"status {a.status} vs {b.status}")
            return
        run_a = _redirect_path(a.location).rstrip("/").split("/")[-1]
        run_b = _redirect_path(b.location).rstrip("/").split("/")[-1]
        self.norms["baseline"].alias(run_a, "<RUN>")
        self.norms["candidate"].alias(run_b, "<RUN>")
        time.sleep(3)
        steps = [
            ("abort save", "POST", "/abort", {"disposition": "save"}),
            ("resume", "POST", "/resume", {}),
            ("restart", "POST", "/restart", {}),
        ]
        for name, method, suffix, data in steps:
            if name != "abort save":
                self.wait_both(f"/runs/{run_a}/status", f"/runs/{run_b}/status")
            if name == "restart":
                time.sleep(1)
            self.both(method, f"/runs/{run_a}{suffix}", data, label=f"lifecycle {name}",
                      path_candidate=f"/runs/{run_b}{suffix}")
            states = [json.loads(side.get(f"/runs/{run}/status").body).get("state")
                      for side, run in ((self.baseline, run_a), (self.candidate, run_b))]
            print(f"   {name}: {states}", flush=True)
        status_a, status_b = self.wait_both(f"/runs/{run_a}/status", f"/runs/{run_b}/status")
        self.compare_json("lifecycle final status", status_a, status_b)
        self.compare_run("lifecycle", run_a, run_b)
        for side, run in ((self.baseline, run_a), (self.candidate, run_b)):
            side.post("/runs/gratings", {"action": "delete", "selected_run_id": run, "confirm_delete": "1"})
        gone = [not (side.data / "runs" / run).exists() for side, run in ((self.baseline, run_a), (self.candidate, run_b))]
        if gone != [True, True]:
            self.report.diff("lifecycle delete", f"run directory removed: {gone}")
        else:
            self.report.ok()

    def plots(self) -> None:
        """Re-save every saved plot through its edit form, then delete one."""
        print("== plot edit / delete", flush=True)
        plot_ids = sorted(path.name for path in (self.baseline.data / "plots").glob("*/"))
        for plot_id in plot_ids:
            page = self.baseline.get(f"/plots/{plot_id}/edit")
            forms = [form for form in page_forms(page.body.decode("utf-8", "replace"))
                     if form["method"] == "post" and form["fields"]]
            if not forms:
                continue
            self.both("POST", f"/plots/{plot_id}/edit", forms[0]["fields"], label=f"edit plot {plot_id}")
            self.both("GET", f"/plots/{plot_id}", label=f"plot {plot_id} after edit")
            compare_dirs(f"plot {plot_id} dir", self.baseline.data / "plots" / plot_id,
                         self.candidate.data / "plots" / plot_id, self.norms["baseline"], self.norms["candidate"],
                         self.report)
        if plot_ids:
            self.both("POST", f"/plots/{plot_ids[0]}/delete", {}, label="delete plot")
            self.both("GET", "/plots", label="plots after delete")

    # -- report ------------------------------------------------------------------

    def write_report(self) -> Path:
        path = self.work / "parity-report.md"
        lines = [
            "# Web parity report", "",
            f"- baseline: `{self.baseline.src}`", f"- candidate: `{self.candidate.src}`",
            f"- database copy of: `{self.database}`", f"- tier: {self.args.tier}", "",
            f"**{self.report.equal} comparisons equal, {len(self.report.differences)} different, "
            f"{len(self.report.notes)} notes.**", "",
        ]
        if self.report.differences:
            lines += ["## Differences", ""]
            for label, detail in self.report.differences:
                lines += [f"### {label}", "", "```", detail, "```", ""]
        if self.report.notes:
            lines += ["## Notes", ""] + [f"- {note}" for note in self.report.notes]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def run(self) -> int:
        self.prepare()
        self.start()
        phases = [self.crawl, self.endpoints, self.previews]
        if self.args.tier in {"quick", "full"}:
            phases += [lambda: self.replay_grating_runs(full=False), self.replay_plane_mirror_runs,
                       self.replay_multilayer, self.lifecycle, self.plots]
        if self.args.tier == "full":
            phases.append(lambda: self.replay_grating_runs(full=True))
        try:
            for phase in phases:
                phase()
                print(f"   so far: {self.report.equal} equal, {len(self.report.differences)} different",
                      flush=True)
        finally:
            self.stop()
            path = self.write_report()
        print(f"\n{self.report.equal} equal, {len(self.report.differences)} different, "
              f"{len(self.report.notes)} notes -> {path}")
        return 1 if self.report.differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-src", required=True, help="src directory of the reference version")
    parser.add_argument("--candidate-src", default="src", help="src directory of the version under test")
    parser.add_argument("--database", default=".grax-web", help="database folder to copy (never modified)")
    parser.add_argument("--work", required=True, help="scratch folder for the two database copies and the report")
    parser.add_argument("--tier", choices=("read", "quick", "full"), default="quick")
    parser.add_argument("--port", type=int, default=5061, help="baseline port; the candidate uses port + 1")
    parser.add_argument("--points", type=int, default=3, help="points per scan axis in the quick tier")
    parser.add_argument("--full-max-points", type=int, default=150, help="largest saved run replayed at full size")
    parser.add_argument("--rtol", type=float, default=1e-9, help="tolerance against originally stored results")
    return Parity(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
