"""Process memory helpers for simulation logging."""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path


def current_process_memory_bytes() -> int | None:
    """Return current RSS memory usage for the active process in bytes.

    Returns:
        Resident set size in bytes, or ``None`` when it cannot be detected on
        the current platform.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    if sys.platform.startswith("win"):
        try:

            class _ProcessMemoryCounters(ctypes.Structure):
                """Windows process memory counters layout."""

                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process_handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.WorkingSetSize)
        except Exception:
            pass

    return None


def format_memory_mb(memory_bytes: int) -> str:
    """Return a stable megabyte string for a byte count.

    Args:
        memory_bytes: Memory value in bytes.

    Returns:
        Memory formatted in mebibytes with two decimal places.
    """
    return f"{float(memory_bytes) / (1024.0 * 1024.0):.2f} MB"


class PeakMemorySampler:
    """Sample process RSS in the background during one simulation call."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        """Initialize an inactive sampler.

        Args:
            interval_seconds: Delay between RSS samples while the context is active.
        """
        self.interval_seconds = max(float(interval_seconds), 0.001)
        self.baseline_memory_bytes: int | None = None
        self.peak_memory_bytes: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakMemorySampler:
        """Start sampling process RSS when an initial sample is available."""
        initial_memory = current_process_memory_bytes()
        if initial_memory is None:
            return self

        self.baseline_memory_bytes = int(initial_memory)
        self.peak_memory_bytes = int(initial_memory)
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="grax-peak-memory-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Stop sampling and capture one final RSS sample."""
        self._update_peak()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2.0)
        self._update_peak()

    @property
    def memory_delta_bytes(self) -> int | None:
        """Return peak RSS minus baseline RSS in bytes."""
        if self.baseline_memory_bytes is None or self.peak_memory_bytes is None:
            return None
        return max(0, int(self.peak_memory_bytes) - int(self.baseline_memory_bytes))

    def _sample_until_stopped(self) -> None:
        """Sample RSS until the context exits."""
        while not self._stop_event.wait(self.interval_seconds):
            self._update_peak()

    def _update_peak(self) -> None:
        """Refresh the stored peak RSS from the current process RSS."""
        current_memory = current_process_memory_bytes()
        if current_memory is None:
            return
        if self.peak_memory_bytes is None or current_memory > self.peak_memory_bytes:
            self.peak_memory_bytes = int(current_memory)
