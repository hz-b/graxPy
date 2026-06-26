"""Global worker-pool accounting for the local simulation server."""

import os
import threading

OS_CORE_RESERVE: int = 2

_condition = threading.Condition(threading.Lock())
_allocated: dict[str, int] = {}  # run_id → granted workers


def _pool_total() -> int:
    return max(1, (os.cpu_count() or 4) - OS_CORE_RESERVE)


def allocate_workers(run_id: str) -> int:
    """Block until pool has capacity, then reserve and return granted count.

    Blocks if the global pool is fully allocated. Always grants at least 1 worker.
    The run remains in "queued" state until this returns.
    """
    with _condition:
        while True:
            total = _pool_total()
            free = total - sum(_allocated.values())
            if free >= 1:
                break
            _condition.wait(timeout=10.0)
        _allocated[run_id] = free
    return free


def release_workers(run_id: str) -> None:
    """Release workers held by one run and wake any waiting threads."""
    with _condition:
        _allocated.pop(run_id, None)
        _condition.notify_all()


def resource_status() -> dict:
    """Return current pool utilisation for the /system/resource-status endpoint."""
    total = _pool_total()
    with _condition:
        used = sum(_allocated.values())
    return {
        "cpu_count": os.cpu_count() or 4,
        "os_core_reserve": OS_CORE_RESERVE,
        "total_available": total,
        "pool_used": used,
        "pool_free": max(0, total - used),
        "active_runs": dict(_allocated),
    }
