# Activate Numba Backend In Scripts

Install the optional dependency first:

```bash
python -m pip install "graxpy[numba]"
```

Enable the optional backend in solver calls:

```python
import grax

result = grax.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    backend="numba",
)
```

```{note}
The numpy backend is now the default for `run_simulation`, providing reliable
performance without dependencies. For maximum speed, use `backend="numba"`
which provides 3.7x speedup with identical numerical results.
```

```{important}
If Numba is unavailable, requesting `backend="numba"` raises a clear error.
```
