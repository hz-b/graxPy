"""Thread-pool control for the native linear-algebra libraries.

The Nevière differential-method solver issues thousands of small dense
``solve`` / ``@`` calls per photon-energy point (one interface-response block
per z-slice, plus the sub-block cascade). With a threaded BLAS -- OpenBLAS in
particular -- driving many-core machines, those tiny ``zgesv`` / ``zgemm`` calls
spend most of their time in thread dispatch, and some OpenBLAS builds crash
outright (SIGSEGV in threaded ``zgemm`` / ``getrf``) under that call pattern.

Environment variables such as ``OPENBLAS_NUM_THREADS`` are read only once, when
the BLAS library loads. They cannot limit a library that is already loaded: the
current process in a serial run (``max_workers=1``) or a direct
:func:`grax.run_simulation` call, and the batch workers on Linux, which are
forked from a parent that has already imported NumPy. This module therefore
limits the pools at run time: scoped to one solve with
:func:`single_threaded_blas`, or for a whole worker process with
:func:`limit_blas_threads_for_process`.
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


def limit_blas_threads_for_process() -> None:
    """Limit BLAS/LAPACK to one thread for the rest of this process's life.

    Meant for batch worker initializers: several workers each running a
    multi-threaded BLAS oversubscribe the CPU. Unlike setting environment
    variables, this also works when the library is already loaded, as in a
    forked worker. A no-op where no controllable native library is loaded.
    """

    threadpool_limits(limits=1, user_api="blas")
