# Blazed Grating

This tutorial documents the blazed optimizer workflow in
`examples/optimizer/optimizer_blazed`, implemented via the measurement-fit
optimizer API, for fitting a blazed grating model to measured monochromator
efficiency data.

For API details of optimizer functions/configs, see
[`grax_opt` optimization API](../api/optimization.md).


## Run the Example

Run only the fit step:

```bash
python examples/optimizer/optimizer_blazed/0_fit_blazed_grating.py
```

Run the full workflow:

```bash
./examples/optimizer/optimizer_blazed/run_all.sh
```

The full workflow runs:

1. fit optimization (`0_fit_blazed_grating.py`)
2. simulation with design parameters (`1_run_simulation_design_parameters.py`)
3. simulation with fitted parameters (`2_run_simulation_fitted_parameters.py`)
4. comparison plot generation (`3_plot_blazed_fit_comparison.py`)


## Design vs Optimized Parameters

The example starts from the design parameters in
`examples/optimizer/optimizer_blazed/0_fit_blazed_grating.py`.

| Parameter | Optimized in this example | Design value | Best-fit value |
| --- | --- | --- | --- |
| `period_lpermm` | No (fixed) | `600.0` | `600.0` |
| `blaze_angle_deg` | Yes | `0.729` | `0.8594103033260448` |
| `anti_blaze_angle_deg` | Yes | `5.597` | `2.0` |
| `top_cap_thickness_nm` | Yes | `0.7` | `2.0` |

## Generated figures

```{image} images/optimizer/blazed_fit/best_fit.png
:alt: Measurement versus best-fit curve for blazed optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/blazed_fit/optimization_loss_history.png
:alt: Optimization loss history for blazed optimizer example
:align: center
:width: 80%
```

```{image} images/optimizer/blazed_fit/blazed_fit_measurement_comparison.png
:alt: Design versus fitted versus measured efficiency comparison for blazed example
:align: center
:width: 80%
```
