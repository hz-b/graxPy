# Multiprocessing

`grax.BatchSimulationRunner` supports case-level multiprocessing through
`max_workers`.

## Basic usage

Serial execution:

```python
runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=15,
    max_workers=1,
    show_progress=True,
)
```

Automatic worker selection:

```python
runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=15,
    max_workers="auto",
    show_progress=True,
)
```

Explicit worker count:

```python
runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=15,
    max_workers=8,
    show_progress=True,
)
```

`max_workers` options:

- `1`: always serial
- positive integer: exact worker count
- `"all"`: all logical CPU cores
- `"auto"`: automatic CPU and memory-aware cap

When `max_workers="auto"`, `BatchSimulationRunner` does the following:

1. It computes a CPU-based cap as `max(os.cpu_count() - 2, 1)` (keeps two cores free).
2. It executes one pending case first in the main process (calibration case).
3. It estimates per-worker memory from that run using current RSS memory,
   multiplied by a safety factor (`1.35`).
4. It estimates available system memory and keeps a fixed reserve of `4 GiB`.
5. It computes a memory-based cap from remaining memory and takes:
   `min(cpu_cap, memory_cap)`, with a minimum of `1`.

Fallback behavior:

- If memory availability cannot be detected, `auto` uses the CPU-based cap.
- If there is only one pending case, `auto` uses the CPU-based cap.
- If usable memory after reserve is non-positive, it forces `1` worker.

## Platform note

On Windows, multiprocessing uses `spawn`. If you see worker startup or pickling
issues in your environment, use:

```python
max_workers=1
```

This keeps execution reliable while preserving the same simulation workflow.

For checkpointing, resume, and progress behavior, see
{doc}`checkpoints-and-resume`.
