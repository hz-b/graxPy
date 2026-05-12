# Package organization

The `grax.simulation` module is implemented as a package to separate concerns by responsibility. The public API remains available from `grax.simulation` and `grax`, but the implementation is split across multiple modules.

## Module structure

- `models.py`: Typed result dataclasses (`SingleSimulationResult`, `BatchSimulationResult`, `CaseExecutionResult`, etc.) and small compatibility containers.
- `core.py`: One-point RCWA execution (`run_simulation`, `RCWASimulation`) and legacy plotting/comparison helpers.
- `batch.py`: Generic batch execution (`BatchSimulationRunner`), checkpointing, subprocess execution, and worker calibration helpers.
- `cases.py`: Lazy case-generation helpers (`energy_angle_cases`, `fixed_angle_cases`, `monochromator_cases`) and monochromator angle generation.
- `theta_search.py`: Single-energy multilayer theta-search logic (`run_multilayer_theta_search`) and scan helpers.
- `theta_search_sweep.py`: Multi-energy adaptive theta-search sweep workflow (`run_multilayer_theta_search_sweep`) and artifact writing.
- `serialization.py`: JSON/checkpoint serialization for single and case results.

## Public API

All public symbols are re-exported from `grax.simulation` so existing code continues to work. Several private helpers are also re-exported because the existing test suite patches them directly.

## Internal dispatch

Internal execution paths that used to read module globals from the flat module now dispatch through the package facade where needed so monkeypatch-based tests continue to work.
