# Resume an optimizer run

Both `grax_opt.optimize_to_measurements` and
`grax_opt.optimize_to_joint_measurements` checkpoint their progress, so a run
that was interrupted can be continued, and a run that finished can be extended
with more trials.

## Extending a finished run

`total_trials` is **cumulative**. To take a completed 100-trial fit out to 1000
trials, raise the number and set `resume=True`:

```python
result = optimize_to_measurements({
    ...,
    "total_trials": 1000,   # was 100
    "resume": True,
})
```

This runs 900 more trials and keeps the first 100. It does **not** re-run them,
and the restored Ax client keeps its surrogate model, so the new trials continue
to benefit from everything the first run learned.

If `total_trials` is already reached, the run is a no-op: the artifacts are
rewritten and the existing result is returned.

## Recovering an interrupted run

Checkpoints are written on **every** run, whether or not `resume` is set. If a
run is interrupted, simply rerun the same script with `resume=True` — nothing
has to be prepared in advance.

## Spec keys

- `resume`: read an existing checkpoint before starting. Defaults to `False`.
- `checkpoint_dir`: where checkpoints live. Defaults to `output_dir/checkpoint`.
- `checkpoint_interval`: trials between full checkpoint flushes. Defaults to `1`.
  Raise it for very long runs if snapshot writes become noticeable.

Leaving `resume=True` in a script permanently is safe: when no checkpoint
exists, a new run simply starts.

## Checkpoint contents

`checkpoint_dir` holds three files:

- `ax_client_snapshot.json`: the Ax client state, so the surrogate model
  survives the restart instead of falling back to a fresh random phase.
- `optimizer_state.json`: best-so-far results, counters, timing metadata, and
  the problem fingerprint.
- `trial_records.jsonl`: append-only per-trial history.

The two JSON files are written atomically, so an interrupted write leaves the
previous good checkpoint intact. A partially written final line of
`trial_records.jsonl` is skipped with a warning on the next resume.

## What may and may not change between runs

Resuming re-checks that the optimization problem itself is unchanged. Anything
that would make old trials incomparable to new ones blocks the resume:

| May change freely | Blocks the resume |
| --- | --- |
| `total_trials`, `batch_size` | `parameter_bounds`, `equality_constraints` |
| `max_workers`, `backend` | `diffraction_order`, `fourier_orders`, `roughness_sigma_nm` |
| `save_best_fit_plot`, `save_loss_plot` | the measurement files' **contents** or paths |
| early-stopping settings | evaluation energies and angles, `objective_name` |
| `checkpoint_interval`, `output_dir` | `joint_loss_reduction` and per-angle weights |

Measurements are identified by content hash, so editing a `.dat` file blocks the
resume even when the path is unchanged. When something does differ, the error
names the offending keys:

```text
ValueError: Cannot resume: the checkpoint was created for a different
optimization problem (changed: parameter_bounds.depth_nm). Use a different
checkpoint_dir or set resume=False to start a new run.
```

`random_seed` is recorded but does not block a resume; changing it only affects
trials that have not been generated yet.

## Failure handling

- **No checkpoint present**: starts a new run and logs that it did so.
- **Incomplete checkpoint** (state or snapshot missing): raises, rather than
  silently discarding the other half.
- **Unreadable Ax snapshot**: raises, reporting the Ax version the checkpoint
  was written with alongside the installed one. Use a different
  `checkpoint_dir`, or `resume=False`, to start over.
- **Different Ax version**: warns and attempts the load.

A resume never silently throws away recorded progress; it either continues or
tells you why it cannot.
