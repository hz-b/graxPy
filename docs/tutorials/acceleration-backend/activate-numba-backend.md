# Activate Numba Backend In Scripts

Install the optional dependency first:

```bash
python -m pip install "graxpy[numba]"
```

Enable the optional backend in solver calls:

```python
import grax as rp

result = rp.run_simulation(
    grating=grating,
    energy_ev=energy_ev,
    grazing_angle_deg=grazing_angle_deg,
    diffraction_order=diffraction_order,
    fourier_orders=fourier_orders,
    _fourier_backend="numba-optional",
)
```

```{important}
If Numba is unavailable, `numba-optional` automatically falls back to the
baseline backend, and summary output reports that fallback explicitly.
```
