"""Test cases for the __main__ module."""
from pathlib import Path

from laptrack import __main__
from laptrack import LapTrack


def test_tap_configure():
    args = __main__.__TrackArgs().parse_args(
        "--coordinate_cols position_x position_y --csv_path test.csv --output_path test.zarr --metric sqeuclidean --cutoff 255".split()
    )
    lt_kwargs = {name: getattr(args, name) for name in LapTrack.model_fields}
    lt = LapTrack(**lt_kwargs)
    assert args.csv_path == Path("test.csv")
    assert args.output_path == Path("test.zarr")
    assert args.coordinate_cols == ["position_x", "position_y"]
    assert args.frame_col == "frame"
    assert lt.metric == "sqeuclidean"
    assert lt.cutoff == 255
    assert lt.splitting_cutoff
    print(lt)
