"""Test cases for the __main__ module."""

from os import path

import numpy as np
import pandas as pd
import pytest
from skimage.io import imsave

from laptrack import __main__
from laptrack import LapTrack
from laptrack import OverLapTrack


@pytest.fixture
def spots_csv_path(shared_datadir, tmp_path):
    df = pd.read_csv(
        path.join(shared_datadir, "trackmate_tracks_with_splitting_spots.csv")
    )
    csv_path = tmp_path / "spots.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def labels():
    # four frames to avoid the tiff RGB interpretation of three-plane stacks
    labels = np.zeros((4, 20, 20), dtype=np.uint16)
    # two objects moving to the lower right
    for frame in range(4):
        labels[frame, frame : frame + 3, frame : frame + 3] = 1
        labels[frame, frame + 10 : frame + 13, frame + 8 : frame + 11] = 2
    return labels


def test_track_args_parse() -> None:
    args = __main__.TrackArgs().parse_args(
        (
            "--csv_path test.csv --output_csv_dir out "
            "--coordinate_cols position_x position_y "
            "--metric sqeuclidean --cutoff 255 --gap_closing_cutoff false "
            "--track_start_cost 10"
        ).split()
    )
    lt = LapTrack(**__main__._model_kwargs_from_args(args, LapTrack))
    assert args.coordinate_cols == ["position_x", "position_y"]
    assert args.frame_col == "frame"
    assert lt.metric == "sqeuclidean"
    assert lt.cutoff == 255
    assert lt.gap_closing_cutoff is False
    assert lt.track_start_cost == 10
    assert not lt.splitting_cutoff


def test_overlap_track_args_parse() -> None:
    args = __main__.OverLapTrackArgs().parse_args(
        (
            "--labels_path labels.tif --output_csv_dir out "
            "--metric_coefs 1 -1 0 0 0 --cutoff 0.9"
        ).split()
    )
    olt = OverLapTrack(**__main__._model_kwargs_from_args(args, OverLapTrack))
    assert olt.metric_coefs == (1.0, -1.0, 0.0, 0.0, 0.0)
    assert olt.cutoff == 0.9
    # the metric fields are not exposed in the command line
    assert not hasattr(args, "metric")
    assert not hasattr(args, "splitting_metric")


def test_run_track(tmp_path, spots_csv_path) -> None:
    csv_path, df = spots_csv_path
    out_dir = tmp_path / "out"
    geff_path = tmp_path / "tracks.geff"
    __main__.main(
        (
            f"track --csv_path {csv_path} --output_csv_dir {out_dir} "
            f"--output_geff_path {geff_path} "
            "--coordinate_cols position_x position_y "
            "--cutoff 225 --splitting_cutoff 225"
        ).split()
    )
    track_df = pd.read_csv(out_dir / "track.csv")
    split_df = pd.read_csv(out_dir / "split.csv")
    pd.read_csv(out_dir / "merge.csv")

    lt = LapTrack(cutoff=225, splitting_cutoff=225)
    track_df2, split_df2, _merge_df2 = lt.predict_dataframe(
        df, ["position_x", "position_y"]
    )
    assert len(track_df) == len(track_df2)
    assert set(track_df["track_id"]) == set(track_df2["track_id"])
    assert len(split_df) == len(split_df2)

    geff = pytest.importorskip("geff")
    graph, _metadata = geff.read(geff_path)
    assert graph.number_of_nodes() == len(df)


def test_run_overlap_track(tmp_path, labels) -> None:
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    out_dir = tmp_path / "out"
    __main__.main(
        (
            f"overlap_track --labels_path {labels_path} "
            f"--output_csv_dir {out_dir} "
            "--metric_coefs 1 0 -1 0 0 --cutoff 0.99"
        ).split()
    )
    track_df = pd.read_csv(out_dir / "track.csv")
    assert set(track_df["label"]) == {1, 2}
    # each object is tracked over all frames as a single track
    assert track_df.groupby("label")["track_id"].nunique().max() == 1
    assert track_df["track_id"].nunique() == 2


def test_read_labels(tmp_path, labels) -> None:
    zarr = pytest.importorskip("zarr")

    np.save(tmp_path / "labels.npy", labels)
    imsave(tmp_path / "labels.tif", labels, check_contrast=False)
    zarr.save(str(tmp_path / "labels.zarr"), labels)
    tiff_dir = tmp_path / "labels"
    tiff_dir.mkdir()
    for frame, label in enumerate(labels):
        imsave(tiff_dir / f"labels_{frame:04d}.tif", label, check_contrast=False)

    for p in ["labels.npy", "labels.tif", "labels.zarr", "labels"]:
        assert (labels == __main__._read_labels(tmp_path / p)).all(), p

    with pytest.raises(ValueError):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        __main__._read_labels(empty_dir)


def test_main_requires_output(tmp_path, spots_csv_path) -> None:
    csv_path, _df = spots_csv_path
    with pytest.raises(ValueError):
        __main__.main(
            (
                f"track --csv_path {csv_path} --coordinate_cols position_x position_y"
            ).split()
        )


def test_main_no_subcommand() -> None:
    with pytest.raises(SystemExit):
        __main__.main([])
