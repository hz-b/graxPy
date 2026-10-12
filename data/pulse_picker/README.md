# Fig. 4 digitization provenance

Source: S. Vadilonga *et al.*, “Piezo-modulated active grating for selecting X-ray pulses separated by one nanosecond,” *Optics Express* **29**, 34962 (2021), [DOI:10.1364/OE.438570](https://doi.org/10.1364/OE.438570).

`paper_fig4.png` is the first raster image on PDF page 7, extracted by `tools/pulse_picker/extract_figure.py`. It preserves the original two-panel raster figure. `fig4_digitized.csv` is produced by `tools/pulse_picker/digitize_fig4.py` from its left (linear-scale) panel. The CSV in `src/grax/pulse_picker_data/` is the identical packaged copy used by the web application.

The pixel calibration is in the digitizer source: zero angle at x=370 px, 0.04° across 555 px, zero intensity at y=436 px, and unit intensity at y=23 px. Points are sampled every 5 px; estimated reading uncertainty is ±2 px horizontally and ±3 px vertically. Trace colors, median-pixel extraction, and legend exclusion are documented in the script. These are **raster-derived estimates**, not the authors' raw data. Overlapping traces, antialiasing, and the line width add unquantified systematic uncertainty. The source panel is retained for visual audit.

The model's 0.85 eV effective bandwidth and 0.0104 nm/V effective displacement response were fitted to these raster points. Neither value is asserted to be the paper's measured instrument bandwidth or true piezoelectric strain coefficient.
