"""ASV benchmarks for the frame-to-frame linking."""

import numpy as np

from laptrack import LapTrack


class LinkingSuite:
    """Benchmark the linking step for the different computation modes."""

    timeout = 600
    params = ([300, 3000, 10000], ["baseline", "kdtree", "gpu"])
    param_names = ["n_points", "mode"]

    def setup(self, n_points, mode):
        """Prepare the coordinates and the tracker."""
        rng = np.random.RandomState(0)
        self.coords = [rng.random((n_points, 2)) * 1000 for _ in range(5)]
        kwargs = dict(
            cutoff=15.0**2,
            gap_closing_cutoff=False,
            splitting_cutoff=False,
            merging_cutoff=False,
        )
        if mode == "kdtree":
            kwargs["metric_is_distance"] = True
        elif mode == "gpu":
            try:
                import cupy  # noqa: F401
            except ImportError:
                raise NotImplementedError("cupy is not installed")
            kwargs["use_gpu"] = True
        self.lt = LapTrack(**kwargs)

    def time_predict(self, n_points, mode):
        """Time the prediction."""
        self.lt.predict(self.coords)


class GapSplitMergeSuite:
    """Benchmark the tracking with gap closing, splitting and merging."""

    timeout = 600
    params = [300, 1000]
    param_names = ["n_points"]

    def setup(self, n_points):
        """Prepare the coordinates and the tracker."""
        rng = np.random.RandomState(0)
        self.coords = [rng.random((n_points, 2)) * 1000 for _ in range(5)]
        self.lt = LapTrack(
            cutoff=15.0**2,
            gap_closing_cutoff=15.0**2,
            splitting_cutoff=15.0**2,
            merging_cutoff=15.0**2,
        )

    def time_predict(self, n_points):
        """Time the prediction."""
        self.lt.predict(self.coords)
