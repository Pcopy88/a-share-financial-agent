"""Sample process RSS; observe only an already initialized local CUDA device."""

import sys
import threading
import time

import psutil


class ResourceSampler:
    """50 ms RSS sampling, plus torch allocator peaks for the current CUDA device.

    Does not import torch, initialize CUDA, or inspect cloud VRAM. CUDA peak reset
    affects this process's current device statistics; use around an isolated run.
    RSS samples can miss shorter spikes; torch numbers exclude non-torch GPU memory.
    """

    def __init__(self, interval_seconds=0.05, *, process=None):
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.process = process if process is not None else psutil.Process()
        self.result = {"peak_rss_bytes": None, "rss_sample_count": 0,
                       "rss_sampling_interval_seconds": interval_seconds,
                       "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
                       "gpu_device": None, "cloud_gpu_memory_bytes": None,
                       "gpu_scope": "current local CUDA device torch allocator only",
                       "errors": []}
        self._stop = threading.Event()
        self._thread = None
        self._cuda = None
        self._used = False

    def _rss(self):
        try:
            rss = self.process.memory_info().rss
            previous = self.result["peak_rss_bytes"]
            self.result["peak_rss_bytes"] = max(previous, rss) if previous is not None else rss
            self.result["rss_sample_count"] += 1
        except (psutil.Error, OSError) as exc:
            self.result["errors"].append("RSS:" + type(exc).__name__)

    def _sample(self):
        while not self._stop.wait(self.interval_seconds):
            self._rss()

    def __enter__(self):
        if self._used:
            raise RuntimeError("ResourceSampler instances are single-use")
        self._used = True
        self._started = time.perf_counter()
        self._rss()
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_initialized():
                device = cuda.current_device()
                cuda.reset_peak_memory_stats(device)
                self._cuda = cuda
                self.result["gpu_device"] = device
        except (RuntimeError, AttributeError) as exc:
            self.result["errors"].append("CUDA:" + type(exc).__name__)
        self._thread = threading.Thread(target=self._sample, daemon=True, name="rss-sampler")
        self._thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._stop.set()
        self._thread.join(timeout=1)
        self._rss()
        if self._cuda is not None:
            try:
                device = self.result["gpu_device"]
                self.result["gpu_peak_allocated_bytes"] = self._cuda.max_memory_allocated(device)
                self.result["gpu_peak_reserved_bytes"] = self._cuda.max_memory_reserved(device)
            except (RuntimeError, AttributeError) as exc:
                self.result["errors"].append("CUDA:" + type(exc).__name__)
        self.result["elapsed_seconds"] = time.perf_counter() - self._started
        return False
