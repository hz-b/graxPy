# Checkpoints and resume

Use checkpointing when a batch run is long or may be interrupted.

## Basic setup

```python
runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=25,
    checkpoint_dir="results/checkpoints",
    checkpoint_interval=1,
    resume=True,
    show_progress=True,
)
```

Key parameters:

- `checkpoint_dir`: output folder for checkpoint files
- `checkpoint_interval`: flush frequency for completed case records
- `resume=True`: skip already completed case IDs from prior runs

## Progress behavior with resume

Progress updates when a case is completed, and also advances for cases skipped
because they already exist in the checkpoint.

This means a resumed run can start with immediate progress movement while
previously completed case IDs are filtered out.

## Restart workflow

1. Keep case ordering stable across reruns (or provide explicit stable `case_id` values).
2. Reuse the same `checkpoint_dir`.
3. Run again with `resume=True`.
4. Only missing cases are recomputed.

`case_id` is optional. When omitted, the runner generates deterministic IDs
from workflow and index.

## Failure/recovery notes

- If a run is interrupted, rerun with the same checkpoint settings.
- If you intentionally want a clean rerun, use a different `checkpoint_dir` or
  remove the old checkpoint artifacts first.
- Keep checkpointing settings close to your output CSV workflow to avoid mixing
  runs from unrelated studies.
