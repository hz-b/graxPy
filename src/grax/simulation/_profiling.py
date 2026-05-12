"""Internal lightweight profiling helpers for single RCWA solves."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
import tracemalloc


@dataclass(frozen=True)
class StageTiming:
    """Aggregate timing entry for one stage."""

    stage: str
    seconds_inclusive: float
    seconds_exclusive: float
    percent_exclusive: float
    calls: int


class SolverProfiler:
    """Collect lightweight stage timings for one solver run.

    This class is internal and intended for one-case RCWA profiling only.
    """

    def __init__(self) -> None:
        """Initialize empty timing and counter registries."""

        self._stage_seconds: dict[str, float] = defaultdict(float)
        self._stage_exclusive_seconds: dict[str, float] = defaultdict(float)
        self._stage_calls: dict[str, int] = defaultdict(int)
        self._counters: dict[str, int] = defaultdict(int)
        self._active_stack: list[list[float | str]] = []
        self._run_start: float = perf_counter()
        self._run_end: float | None = None
        self._tracked_memory: bool = False
        self._peak_memory_bytes: int = 0
        self._stage_peak_memory_bytes: dict[str, int] = defaultdict(int)
        self._detail_lists: dict[str, list[float]] = defaultdict(list)
        self._detail_counts: dict[str, int] = defaultdict(int)
        self._detail_peaks: dict[str, float] = {}
        self._unique_sets: dict[str, set[str]] = defaultdict(set)
        self._metadata: dict[str, object] = {}

    @contextmanager
    def record(self, stage: str):
        """Record elapsed time spent in one named stage."""

        start = perf_counter()
        frame: list[float | str] = [stage, start, 0.0]
        self._active_stack.append(frame)
        try:
            yield
        finally:
            end = perf_counter()
            elapsed = end - start
            child_elapsed = float(frame[2])
            exclusive_elapsed = max(0.0, elapsed - child_elapsed)

            self._active_stack.pop()
            if self._active_stack:
                self._active_stack[-1][2] = float(self._active_stack[-1][2]) + elapsed

            self._stage_seconds[stage] += elapsed
            self._stage_exclusive_seconds[stage] += exclusive_elapsed
            self._stage_calls[stage] += 1
            self._update_memory(stage)

    def increment(self, name: str, value: int = 1) -> None:
        """Increment an integer counter."""

        self._counters[name] += int(value)

    def add_detail_timing(self, name: str, seconds: float) -> None:
        """Store one detailed timing sample for an internal operation."""

        self._detail_lists[name].append(float(seconds))

    def add_detail_count(self, name: str, value: int = 1) -> None:
        """Accumulate one detailed integer counter."""

        self._detail_counts[name] += int(value)

    def add_unique_value(self, name: str, value: str) -> None:
        """Track unique string identifiers for repeated operations."""

        self._unique_sets[name].add(value)

    def update_detail_peak(self, name: str, value: float) -> None:
        """Track the maximum observed value for one detailed metric."""

        current = self._detail_peaks.get(name)
        if current is None or float(value) > current:
            self._detail_peaks[name] = float(value)

    def set_metadata(self, name: str, value: object) -> None:
        """Store run metadata for reproducible benchmark reports."""

        self._metadata[name] = value

    def enable_memory_tracking(self) -> None:
        """Enable lightweight peak memory tracking via ``tracemalloc``."""

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracked_memory = True
        self._update_memory(None)

    def finalize(self) -> None:
        """Stop overall wall-clock timer if not already stopped."""

        if self._run_end is None:
            self._run_end = perf_counter()
        self._update_memory(None)
        if self._tracked_memory and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracked_memory = False

    def _update_memory(self, stage: str | None) -> None:
        """Update global and stage-local peak memory values."""

        if not tracemalloc.is_tracing():
            return
        _current, peak = tracemalloc.get_traced_memory()
        self._peak_memory_bytes = max(self._peak_memory_bytes, int(peak))
        if stage is not None:
            self._stage_peak_memory_bytes[stage] = max(
                self._stage_peak_memory_bytes.get(stage, 0),
                int(peak),
            )

    @property
    def total_wall_seconds(self) -> float:
        """Return total wall-clock duration for the profiled run."""

        end = self._run_end if self._run_end is not None else perf_counter()
        return max(0.0, end - self._run_start)

    def stage_timings(self) -> list[StageTiming]:
        """Return stage timings sorted by descending runtime."""

        total = self.total_wall_seconds
        total_exclusive = float(sum(self._stage_exclusive_seconds.values()))
        entries: list[StageTiming] = []
        for stage, seconds_inclusive in self._stage_seconds.items():
            seconds_exclusive = float(self._stage_exclusive_seconds.get(stage, 0.0))
            percent_exclusive = (
                0.0 if total_exclusive <= 0.0 else (seconds_exclusive / total_exclusive) * 100.0
            )
            entries.append(
                StageTiming(
                    stage=stage,
                    seconds_inclusive=float(seconds_inclusive),
                    seconds_exclusive=seconds_exclusive,
                    percent_exclusive=float(percent_exclusive),
                    calls=int(self._stage_calls.get(stage, 0)),
                )
            )
        entries.sort(key=lambda item: item.seconds_exclusive, reverse=True)
        return entries

    def summary_dict(self) -> dict[str, object]:
        """Return structured timing summary data."""

        stage_entries = self.stage_timings()
        return {
            "total_wall_seconds": self.total_wall_seconds,
            "stages": [
                {
                    "stage": entry.stage,
                    "seconds_inclusive": entry.seconds_inclusive,
                    "seconds_exclusive": entry.seconds_exclusive,
                    "percent_exclusive": entry.percent_exclusive,
                    "calls": entry.calls,
                }
                for entry in stage_entries
            ],
            "profiled_exclusive_seconds": float(sum(self._stage_exclusive_seconds.values())),
            "unprofiled_seconds": max(
                0.0,
                self.total_wall_seconds - float(sum(self._stage_exclusive_seconds.values())),
            ),
            "counters": dict(self._counters),
            "peak_memory_bytes": int(self._peak_memory_bytes),
            "stage_peak_memory_bytes": dict(self._stage_peak_memory_bytes),
            "metadata": dict(self._metadata),
            "derived_kpis": self._derived_kpis(),
            "details": self._details_summary(),
        }

    def _derived_kpis(self) -> dict[str, float]:
        """Return derived solver KPIs for the current profiling run."""

        kpis: dict[str, float] = {}
        fourier_seconds = float(self._stage_exclusive_seconds.get("fourier_coefficients", 0.0))
        fourier_calls = int(self._detail_counts.get("fourier_calls", 0))
        harmonics = int(self._detail_counts.get("fourier_harmonics_total", 0))
        allocation_bytes = int(self._detail_counts.get("fourier_allocation_bytes", 0))
        phase_cache_peak = float(self._detail_peaks.get("fourier_phase_table_cache_bytes_peak", 0.0))

        if fourier_calls > 0:
            kpis["time_per_fourier_call_seconds"] = fourier_seconds / float(fourier_calls)
            kpis["bytes_per_fourier_call"] = float(allocation_bytes) / float(fourier_calls)
        if harmonics > 0:
            kpis["time_per_harmonic_seconds"] = fourier_seconds / float(harmonics)
        if phase_cache_peak > 0.0:
            for workers in (2, 4, 8):
                kpis[f"phase_table_cache_proxy_mb_{workers}w"] = (
                    phase_cache_peak * float(workers) / (1024.0 * 1024.0)
                )
        return kpis

    def _details_summary(self) -> dict[str, object]:
        """Return aggregated detailed diagnostics."""

        detail_timings: dict[str, dict[str, float]] = {}
        for name, values in self._detail_lists.items():
            if not values:
                continue
            total = float(sum(values))
            detail_timings[name] = {
                "calls": float(len(values)),
                "total_seconds": total,
                "mean_seconds": total / float(len(values)),
                "max_seconds": float(max(values)),
            }
        return {
            "timings": detail_timings,
            "counts": dict(self._detail_counts),
            "peaks": dict(self._detail_peaks),
            "unique_counts": {name: len(values) for name, values in self._unique_sets.items()},
        }

    def format_report(self) -> str:
        """Return a concise human-readable timing report."""

        entries = self.stage_timings()
        lines = [
            "RCWA Profiling Summary",
            f"total_wall_seconds={self.total_wall_seconds:.6f}",
        ]
        if self._metadata:
            lines.extend(["", "metadata"])
            for name in sorted(self._metadata):
                lines.append(f"- {name}: {self._metadata[name]}")
        lines.extend(
            [
                "",
                "stage                      excl_s    excl_%    incl_s   calls",
                "-------------------------------------------------------------",
            ]
        )
        if not entries:
            lines.append("(no recorded stages)")
        for entry in entries:
            lines.append(
                f"{entry.stage:<26} {entry.seconds_exclusive:>8.6f}  {entry.percent_exclusive:>7.2f}%  "
                f"{entry.seconds_inclusive:>8.6f}  {entry.calls:>5d}"
            )
        profiled_exclusive = float(sum(self._stage_exclusive_seconds.values()))
        unprofiled = max(0.0, self.total_wall_seconds - profiled_exclusive)
        lines.extend(
            [
                "",
                f"profiled_exclusive_seconds={profiled_exclusive:.6f}",
                f"unprofiled_seconds={unprofiled:.6f}",
            ]
        )
        derived_kpis = self._derived_kpis()
        if derived_kpis:
            lines.extend(["", "derived_kpis"])
            for name in sorted(derived_kpis):
                lines.append(f"- {name}: {derived_kpis[name]:.6f}")
        if self._counters:
            lines.extend(["", "counters"])
            for name in sorted(self._counters):
                lines.append(f"- {name}: {self._counters[name]}")
        if self._peak_memory_bytes > 0:
            lines.extend(
                [
                    "",
                    "memory",
                    f"- peak_memory_mb: {self._peak_memory_bytes / (1024.0 * 1024.0):.3f}",
                ]
            )
            for stage in sorted(self._stage_peak_memory_bytes):
                peak = self._stage_peak_memory_bytes[stage] / (1024.0 * 1024.0)
                lines.append(f"- stage_peak_mb[{stage}]: {peak:.3f}")
        details = self._details_summary()
        timing_details = details["timings"]
        if timing_details:
            lines.extend(["", "detail_timings"])
            for name in sorted(timing_details):
                item = timing_details[name]
                lines.append(
                    f"- {name}: calls={int(item['calls'])}, total_s={item['total_seconds']:.6f}, "
                    f"mean_s={item['mean_seconds']:.6f}, max_s={item['max_seconds']:.6f}"
                )
        count_details = details["counts"]
        if count_details:
            lines.extend(["", "detail_counts"])
            for name in sorted(count_details):
                lines.append(f"- {name}: {count_details[name]}")
        peak_details = details["peaks"]
        if peak_details:
            lines.extend(["", "detail_peaks"])
            for name in sorted(peak_details):
                lines.append(f"- {name}: {peak_details[name]:.6f}")
        unique_details = details["unique_counts"]
        if unique_details:
            lines.extend(["", "detail_unique_counts"])
            for name in sorted(unique_details):
                lines.append(f"- {name}: {unique_details[name]}")
        return "\n".join(lines)
