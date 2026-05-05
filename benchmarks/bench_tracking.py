"""Performance benchmark harness for laptrack.

Generates synthetic 2D random-walk trajectories with occasional missed
detections (so the gap-closing path is actually exercised), times
``LapTrack().predict()`` across a few feature configurations and dataset
sizes, and writes the results to a CSV.

Usage
-----

    # default: small + medium scenarios across all three configs
    python benchmarks/bench_tracking.py --out benchmarks/baseline.csv

    # add the slow large-scale run
    python benchmarks/bench_tracking.py --scenarios small medium large

    # see per-stage timings on stderr
    python benchmarks/bench_tracking.py --debug-logging 2> bench.log

For comparable numbers across optimization PRs, run on a quiet machine
(no other CPU-heavy processes), pin Python/NumPy/SciPy versions, and
record the commit SHA next to the CSV.

Memory note: ``ru_maxrss`` is a process-wide high-water mark, so the
``peak_rss_mb`` column reflects the largest scenario seen so far in this
process. To get a clean per-scenario number, run with ``--scenarios
<one>`` per invocation.
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np

from laptrack import LapTrack


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def make_random_walk_trajectories(
    n_frames: int,
    n_points: int,
    *,
    box_size: float = 1000.0,
    step_std: float = 3.0,
    drop_prob: float = 0.05,
    seed: int = 0,
) -> List[np.ndarray]:
    """Build a list of (n_points_in_frame, 2) coordinate arrays.

    Each underlying point performs a Gaussian random walk; with probability
    ``drop_prob`` per frame a point is omitted from that frame's output,
    which creates realistic gaps for the gap-closing stage to handle.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, box_size, size=(n_points, 2))
    out: List[np.ndarray] = []
    for _ in range(n_frames):
        coords = coords + rng.normal(scale=step_std, size=coords.shape)
        coords = np.clip(coords, 0, box_size)
        keep = rng.random(n_points) > drop_prob
        out.append(coords[keep].copy())
    return out


SCENARIOS: Dict[str, Dict[str, int]] = {
    "small": dict(n_frames=50, n_points=200),
    "medium": dict(n_frames=200, n_points=2000),
    "large": dict(n_frames=500, n_points=10000),
}


# ---------------------------------------------------------------------------
# tracker configurations
# ---------------------------------------------------------------------------

CUTOFF = 15.0**2  # sqeuclidean cutoff matching LapTrack's defaults


def make_tracker(config: str) -> LapTrack:
    """Build a LapTrack instance for the named feature configuration."""
    if config == "link_only":
        return LapTrack(
            metric="sqeuclidean",
            cutoff=CUTOFF,
            gap_closing_cutoff=False,
            splitting_cutoff=False,
            merging_cutoff=False,
        )
    if config == "gap_closing":
        return LapTrack(
            metric="sqeuclidean",
            cutoff=CUTOFF,
            gap_closing_cutoff=CUTOFF,
            gap_closing_max_frame_count=2,
            splitting_cutoff=False,
            merging_cutoff=False,
        )
    if config == "full":
        return LapTrack(
            metric="sqeuclidean",
            cutoff=CUTOFF,
            gap_closing_cutoff=CUTOFF,
            gap_closing_max_frame_count=2,
            splitting_cutoff=CUTOFF,
            merging_cutoff=CUTOFF,
        )
    raise ValueError(f"unknown config {config!r}")


CONFIGS = ["link_only", "gap_closing", "full"]


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def peak_rss_mb() -> float:
    """High-water-mark resident set size in MB (process-wide, monotonic)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: kilobytes; macOS: bytes
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


@dataclass
class Result:
    """One benchmark measurement row."""

    scenario: str
    config: str
    n_frames: int
    n_points: int
    total_points: int
    wall_seconds: float
    peak_rss_mb: float
    n_nodes: int
    n_edges: int


def run_one(
    scenario: str,
    config: str,
    *,
    n_frames: int,
    n_points: int,
    seed: int,
) -> Result:
    """Generate data, run LapTrack.predict, and time it."""
    coords = make_random_walk_trajectories(n_frames, n_points, seed=seed)
    total_points = sum(c.shape[0] for c in coords)
    tracker = make_tracker(config)

    t0 = time.perf_counter()
    tree = tracker.predict(coords)
    elapsed = time.perf_counter() - t0

    return Result(
        scenario=scenario,
        config=config,
        n_frames=n_frames,
        n_points=n_points,
        total_points=total_points,
        wall_seconds=elapsed,
        peak_rss_mb=peak_rss_mb(),
        n_nodes=tree.number_of_nodes(),
        n_edges=tree.number_of_edges(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_row(r: Result) -> str:
    return (
        f"{r.scenario:<7} {r.config:<12} "
        f"frames={r.n_frames:<5} pts/frame={r.n_points:<6} "
        f"total={r.total_points:<8} "
        f"wall={r.wall_seconds:7.2f}s  rss={r.peak_rss_mb:6.1f}MB  "
        f"nodes={r.n_nodes} edges={r.n_edges}"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="laptrack performance benchmark")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["small", "medium"],
        choices=list(SCENARIOS),
        help="Size scenarios to run. 'large' is slow; opt in explicitly.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIGS,
        choices=CONFIGS,
        help="Feature configurations to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results.csv",
        help="CSV output path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug-logging",
        action="store_true",
        help="Enable laptrack DEBUG logging (per-stage timings on stderr).",
    )
    args = parser.parse_args()

    if args.debug_logging:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    rows: List[Result] = []
    for scenario in args.scenarios:
        params = SCENARIOS[scenario]
        for config in args.configs:
            r = run_one(scenario, config, seed=args.seed, **params)
            rows.append(r)
            print(_format_row(r), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "config",
                "n_frames",
                "n_points",
                "total_points",
                "wall_seconds",
                "peak_rss_mb",
                "n_nodes",
                "n_edges",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.scenario,
                    r.config,
                    r.n_frames,
                    r.n_points,
                    r.total_points,
                    f"{r.wall_seconds:.4f}",
                    f"{r.peak_rss_mb:.2f}",
                    r.n_nodes,
                    r.n_edges,
                ]
            )
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
