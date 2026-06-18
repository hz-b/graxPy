# Backend Selection And NumPy Deprecation

Install `graxpy` normally. The numba backend is part of the standard package:

```bash
python -m pip install graxpy
```

Most user-facing solver calls already default to `backend="numba"`. You can
still pass it explicitly when you want the choice to be visible in a script:

```python
import grax

result = grax.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    polarization="p",
    backend="numba",
)
```

```{note}
The numba backend is now the default for `run_simulation`, batch workflows,
and the multilayer theta-search helpers.
```

```{important}
The legacy `backend="numpy"` path still exists for parity checks and
transition-time comparisons, but it is deprecated and emits a warning when
requested explicitly.
```
