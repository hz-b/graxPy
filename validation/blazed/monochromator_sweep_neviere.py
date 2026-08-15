"""Run the blazed 600 l/mm monochromator sweep with the Nevière differential method.

Thin entry point over ``monochromator_sweep.py``. The grating geometry, energy grid,
resolutions and Fourier truncation stay defined there and only there, so a
change to the physics cannot land in one solver's run and not the other's —
which would quietly make the two incomparable.

Results are written next to the RCWA ones with a ``_neviere`` suffix, so
``comparison_blazed_monochromator_sweep.py`` can overlay both.

Any extra arguments are forwarded, so ``--stride`` and the sweep's own flags
still work:

```bash
python blazed/monochromator_sweep_neviere.py
python blazed/monochromator_sweep_neviere.py --stride 10
```
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SWEEP_SCRIPT = Path(__file__).resolve().parent / "monochromator_sweep.py"

if __name__ == "__main__":
    sys.argv = [str(SWEEP_SCRIPT), "--solver", "neviere", *sys.argv[1:]]
    runpy.run_path(str(SWEEP_SCRIPT), run_name="__main__")
