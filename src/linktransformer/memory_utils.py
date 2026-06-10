import os
import threading
from typing import Optional

import psutil


MB = 1024 ** 2


def current_memory_mb(process: Optional[psutil.Process] = None) -> float:
    process = process or psutil.Process(os.getpid())
    return process.memory_info().rss / MB


class PeakMemoryMonitor:
    """Samples process RSS while a block runs and reports the peak increase."""

    def __init__(self, process: Optional[psutil.Process] = None, interval_seconds: float = 0.001):
        self.process = process or psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.peak_mb = 0.0
        self.peak_delta_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_mb = current_memory_mb(self.process)
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.end_mb = current_memory_mb(self.process)
        self.peak_mb = max(self.peak_mb, self.end_mb)
        self.peak_delta_mb = max(0.0, self.peak_mb - self.start_mb)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, current_memory_mb(self.process))
            self._stop_event.wait(self.interval_seconds)
