"""
Benchmark utilities for the Nvidia PhysicalAI ingestion pipeline.

Collects wall-clock time, CPU time, peak RSS, row counts, and byte sizes
for every pipeline phase.  Results are accumulated in-memory and flushed
to JSON at the end of a run.
"""

import json
import os
import resource
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StepMetrics:
    """Metrics for a single pipeline step."""

    step: str
    table: str
    wall_s: float = 0.0
    cpu_user_s: float = 0.0
    cpu_sys_s: float = 0.0
    peak_rss_mb: float = 0.0
    rows_in: int = 0
    rows_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class BenchmarkTracker:
    """Accumulates per-step metrics and writes a summary JSON."""

    def __init__(self, run_name: str, output_path: Optional[str] = None):
        self.run_name = run_name
        if output_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(module_dir, "benchmark_results.json")
            # Fall back to /tmp if the module directory is not writable
            if not os.access(module_dir, os.W_OK):
                default_path = "/tmp/nvidia_benchmark_results.json"
            output_path = default_path
        self.output_path = output_path
        self.steps: List[StepMetrics] = []
        self._start_wall: float = 0.0
        self._start_cpu: tuple = (0.0, 0.0)
        self._start_rss: float = 0.0
        self._current_step: Optional[str] = None
        self._current_table: Optional[str] = None

    # -- context-manager API for individual steps ----------------------------

    def begin(self, step: str, table: str = ""):
        self._current_step = step
        self._current_table = table
        self._start_wall = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self._start_cpu = (ru.ru_utime, ru.ru_stime)
        self._start_rss = ru.ru_maxrss  # in KB on Linux

    def end(self, rows_in: int = 0, rows_out: int = 0,
            bytes_in: int = 0, bytes_out: int = 0,
            **extra) -> StepMetrics:
        wall = time.perf_counter() - self._start_wall
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user = ru.ru_utime - self._start_cpu[0]
        cpu_sys = ru.ru_stime - self._start_cpu[1]
        peak_rss_mb = ru.ru_maxrss / 1024  # KB → MB

        m = StepMetrics(
            step=self._current_step or "",
            table=self._current_table or "",
            wall_s=round(wall, 3),
            cpu_user_s=round(cpu_user, 3),
            cpu_sys_s=round(cpu_sys, 3),
            peak_rss_mb=round(peak_rss_mb, 1),
            rows_in=rows_in,
            rows_out=rows_out,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            extra=extra,
        )
        self.steps.append(m)
        return m

    # -- summary helpers -----------------------------------------------------

    def total_wall_s(self) -> float:
        return sum(s.wall_s for s in self.steps)

    def total_rows_out(self) -> int:
        return sum(s.rows_out for s in self.steps)

    def flush(self):
        """Write accumulated results to JSON."""
        payload = {
            "run_name": self.run_name,
            "total_wall_s": round(self.total_wall_s(), 3),
            "total_rows_out": self.total_rows_out(),
            "steps": [asdict(s) for s in self.steps],
        }
        # Merge with existing results if the file already has data
        existing: List[Dict] = []
        if os.path.exists(self.output_path):
            with open(self.output_path) as f:
                try:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        existing = [existing]
                except json.JSONDecodeError:
                    existing = []
        existing.append(payload)
        with open(self.output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[BENCH] Results written to {self.output_path}")

    def print_summary(self):
        hdr = f"{'step':<20} {'table':<30} {'wall(s)':>8} {'cpu_u(s)':>9} {'cpu_s(s)':>9} {'RSS(MB)':>9} {'rows_out':>10}"
        print("\n" + "=" * len(hdr))
        print(f"BENCHMARK: {self.run_name}")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for s in self.steps:
            print(
                f"{s.step:<20} {s.table:<30} {s.wall_s:>8.2f} "
                f"{s.cpu_user_s:>9.2f} {s.cpu_sys_s:>9.2f} "
                f"{s.peak_rss_mb:>9.1f} {s.rows_out:>10}"
            )
        print("-" * len(hdr))
        print(
            f"{'TOTAL':<20} {'':<30} {self.total_wall_s():>8.2f} "
            f"{'':>9} {'':>9} {'':>9} {self.total_rows_out():>10}"
        )
        print("=" * len(hdr))
