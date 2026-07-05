"""Command-line interface."""

import sys
from pathlib import Path
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import List
from typing import Literal
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tap import Tap

from . import __version__
from ._tracking import LapTrack
from ._overlap_tracking import OverLapTrack
from ._tracking_result import TrackingResult

#: The fields not exposed as command-line arguments.
_EXCLUDED_FIELDS = {
    LapTrack: [],
    OverLapTrack: [
        # these are overwritten from *_metric_coefs in predict_overlap_dataframe
        "metric",
        "gap_closing_metric",
        "splitting_metric",
        "merging_metric",
    ],
}


def _read_labels(file_path: Union[str, Path]) -> np.ndarray:
    """Read a label image stack from a file, a zarr store, or a directory of tiffs."""
    from tifffile import imread

    file_path = Path(file_path)
    if any(str(file_path).endswith(ext) for ext in [".zarr", ".zarr.zip", ".zr"]):
        import zarr

        return np.asarray(zarr.load(str(file_path)))
    elif file_path.suffix.lower() == ".npy":
        return np.load(file_path)
    elif file_path.is_dir():
        images = []
        for file in sorted(file_path.glob("*.tif")) + sorted(file_path.glob("*.tiff")):
            images.append(imread(file))
        if not images:
            raise ValueError(f"No tiff files found in the directory {file_path}.")
        return np.array(images)
    elif file_path.suffix.lower() in [".tif", ".tiff"]:
        return imread(file_path)
    else:
        from skimage.io import imread as sk_imread

        return sk_imread(file_path)


def _tap_configure_from_cls(clses: List[Type[BaseModel]]) -> Callable:
    """Create a TAP `configure` function adding options for the model fields."""

    def float_or_false(x: str) -> Union[float, bool]:
        if str(x).lower() == "false":
            return False
        return float(x)

    def optional_float(x: Optional[str]) -> Optional[float]:
        if x is None or str(x).lower() == "none":
            return None
        return float(x)

    def is_false_literal(tp: object) -> bool:
        return get_origin(tp) is Literal and get_args(tp) == (False,)

    def configure(self: Tap) -> None:
        for c in clses:
            excluded = _EXCLUDED_FIELDS.get(c, [])
            for name, field in c.model_fields.items():
                if name in excluded:
                    continue
                annotation = field.annotation
                default = field.default

                parser_type: object = annotation
                nargs = None
                if get_origin(annotation) is Union:
                    args = get_args(annotation)
                    if any(a is Callable or a == Callable for a in args):
                        parser_type = str
                    elif any(is_false_literal(a) for a in args) and float in args:
                        parser_type = float_or_false
                    elif float in args and type(None) in args:
                        parser_type = optional_float
                    elif int in args and type(None) in args:
                        parser_type = int
                    elif str in args:
                        parser_type = str
                    else:
                        parser_type = args[0]
                elif get_origin(annotation) is tuple:
                    parser_type = float
                    nargs = len(get_args(annotation))

                kwargs = dict(type=parser_type, default=default, help=field.description)
                if nargs is not None:
                    kwargs["nargs"] = nargs
                self.add_argument(f"--{name}", **kwargs)

    return configure


class TrackArgs(Tap):
    """Track point-like objects listed in a CSV file."""

    csv_path: Path  # Path to the input CSV file with the frame and coordinate columns.
    coordinate_cols: List[str]  # The column names for the coordinates.
    frame_col: str = "frame"  # The column name for the frame index.
    output_csv_dir: Optional[Path] = None
    """Directory to write track.csv, split.csv and merge.csv."""
    output_geff_path: Optional[Path] = None
    """Path of the GEFF zarr store to write the tracks to."""

    configure = _tap_configure_from_cls([LapTrack])


class OverLapTrackArgs(Tap):
    """Track segmentation labels by overlaps."""

    labels_path: Path
    """Path to the input labels (tiff / npy / zarr, or a directory of tiffs)."""
    output_csv_dir: Optional[Path] = None
    """Directory to write track.csv, split.csv and merge.csv."""
    output_geff_path: Optional[Path] = None
    """Path of the GEFF zarr store to write the tracks to."""

    configure = _tap_configure_from_cls([OverLapTrack])


class _Args(Tap):
    """LapTrack: particle tracking by linear assignment problem."""

    def configure(self) -> None:
        self.add_subparsers(dest="cmd", help="Subcommands for laptrack")
        self.add_subparser(
            "track", TrackArgs, help="Track point-like objects from a CSV file."
        )
        self.add_subparser(
            "overlap_track",
            OverLapTrackArgs,
            help="Track segmentation labels by overlaps.",
        )
        self.add_argument("--version", action="version", version=__version__)


def _model_kwargs_from_args(args: Tap, cls: Type[BaseModel]) -> dict:
    excluded = _EXCLUDED_FIELDS.get(cls, [])
    return {
        name: getattr(args, name)
        for name in cls.model_fields
        if name not in excluded and hasattr(args, name)
    }


def _write_result(
    result: TrackingResult,
    output_csv_dir: Optional[Path],
    output_geff_path: Optional[Path],
) -> None:
    if output_csv_dir is None and output_geff_path is None:
        raise ValueError(
            "At least one of --output_csv_dir or --output_geff_path must be provided."
        )
    if output_csv_dir is not None:
        result.write_csvs(output_csv_dir)
    if output_geff_path is not None:
        result.write_geff(output_geff_path)


def track(args: TrackArgs) -> None:
    """Execute point tracking based on the parsed arguments."""
    lt = LapTrack(**_model_kwargs_from_args(args, LapTrack))
    df = pd.read_csv(args.csv_path)
    result = lt.predict_tracking_result(
        df, coordinate_cols=args.coordinate_cols, frame_col=args.frame_col
    )
    _write_result(result, args.output_csv_dir, args.output_geff_path)


def overlap_track(args: OverLapTrackArgs) -> None:
    """Execute overlap tracking based on the parsed arguments."""
    olt = OverLapTrack(**_model_kwargs_from_args(args, OverLapTrack))
    labels = _read_labels(args.labels_path)
    result = olt.predict_overlap_tracking_result(labels)
    _write_result(result, args.output_csv_dir, args.output_geff_path)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the laptrack command."""
    args = _Args().parse_args(argv)
    cmd = getattr(args, "cmd", None)
    if cmd is None:
        print("No subcommand provided. Use 'track' or 'overlap_track'.")
        sys.exit(1)
    if cmd == "track":
        track(args)  # type: ignore[arg-type]
    elif cmd == "overlap_track":
        overlap_track(args)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()  # pragma: no cover
