import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timed_stage(name: str, timings: Dict[str, float]):
    """
    Context manager to record timing of a pipeline stage in milliseconds.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        timings[name] = round(elapsed, 3)
