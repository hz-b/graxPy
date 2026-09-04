"""Thread-pool control for the native linear-algebra libraries.

The Nevière differential-method solver issues thousands of small dense
``solve`` / ``@`` calls per photon-energy point (one interface-response block
per z-slice, plus the sub-block cascade). With a threaded BLAS -- OpenBLAS in
particular -- driving many-core machines, those tiny ``zgesv`` / ``zgemm`` calls
spend most of their time in thread dispatch, and some OpenBLAS builds crash
outright (SIGSEGV in threaded ``zgemm`` / ``getrf``) under that call pattern.

``BatchSimulationRunner`` already pins ``OPENBLAS_NUM_THREADS`` and friends to
``1`` in its spawned workers, but a serial run (``max_workers=1``) or a direct
:func:`grax.run_simulation` call executes in the current process, where those
environment variables were read once at BLAS import and can no longer take
effect. This module limits the pools at run time instead, scoped to the solve.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def single_threaded_blas() -> Iterator[None]:
    """Limit BLAS/LAPACK to one thread for the duration of the block.

    Restores the previous limits on exit. A no-op where no controllable native
    library is loaded (for example NumPy built against Apple Accelerate).
    """

    with threadpool_limits(limits=1, user_api="blas"):
        yield
