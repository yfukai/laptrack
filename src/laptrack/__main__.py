"""Command-line interface."""
import sys
from pathlib import Path
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import geff
import numpy as np
import pandas as pd
import zarr
from skimage.io import imread
from tap import Tap

from . import data_conversion
from . import LapTrack
from . import OverLapTrack

# %%


def _read_image(file_path):
    extension = Path(file_path).suffix.lower()
    if extension in [".zarr", ".zarr.zip", ".zr", ".zr.zip"]:
        return zarr.load(file_path)
    elif extension in [".npy"]:
        return np.load(file_path)
    else:
        if Path(file_path).is_dir():
            images = []
            for file in sorted(Path(file_path).glob("*.tif")):
                images.append(imread(file))
            return np.array(images)
        else:
            return imread(file_path)


def _tap_configure_from_cls(cls: type) -> Callable:
    def configure(self) -> None:
        """Configure parser based on :class:`LapTrack` fields."""

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

        for name, field in LapTrack.model_fields.items():
            annotation = field.annotation
            default = field.default

            parser_type: object = annotation
            if get_origin(annotation) is Union:
                args = get_args(annotation)
                if any(
                    callable_arg is Callable or callable_arg == Callable
                    for callable_arg in args
                ):
                    parser_type = str
                elif any(is_false_literal(a) for a in args) and float in args:
                    parser_type = float_or_false
                elif float in args and type(None) in args:
                    parser_type = optional_float
                elif str in args:
                    parser_type = str
                else:
                    parser_type = args[0]

            self.add_argument(f"--{name}", type=parser_type, default=default)

    return configure


class _TrackArgs(Tap):
    csv_path: Path  # path to input csv file
    output_path: Path
    track_geff_path: Optional[Path] = None  # path to output geff file
    frame_col: str = "frame"
    coordinate_cols: List[str]
    configure = _tap_configure_from_cls(LapTrack)


class _OverLapTrackArgs(Tap):
    labels_path: Path  # path to input csv file
    output_path: Path
    track_geff_path: Optional[Path] = None  # path to output geff file
    configure = _tap_configure_from_cls(OverLapTrack)


def track(args: _TrackArgs) -> None:
    """Execute tracking based on parsed arguments."""
    df = pd.read_csv(args.csv_path)
    tracked_tree = geff.read_nx(args.track_geff_path)
    # FIXME

    lt_kwargs = {name: getattr(args, name) for name in LapTrack.model_fields}
    lt = LapTrack(**lt_kwargs)

    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=args.coordinate_cols, frame_col=args.frame_col
    )

    geff_tree = data_conversion.convert_dataframes_to_geff_networkx(
        track_df,
        split_df,
        merge_df,
        coordinate_cols=args.coordinate_cols,
        frame_col=args.frame_col,
    )
    geff.write_nx(geff_tree, args.output_path)


def track_overlap(args: _OverLapTrackArgs) -> None:
    """Execute overlap tracking based on parsed arguments."""
    labels = _read_image(args.labels_path)
    tracked_tree = geff.read_nx(args.track_geff_path)

    lt_kwargs = {name: getattr(args, name) for name in OverLapTrack.model_fields}
    lt = OverLapTrack(**lt_kwargs)

    track_df, split_df, merge_df = lt.predict_overlap_dataframe(labels)

    geff_tree = data_conversion.convert_dataframes_to_geff_networkx(
        track_df, split_df, merge_df, coordinate_cols=["seg_id"], frame_col="frame"
    )
    geff.write_nx(geff_tree, args.output_path)


def main():  # noqa: D103
    if len(sys.argv) < 2:
        print("Please specify a subcommand: train or test")
        sys.exit(1)

    subcommand = sys.argv[1]
    sub_args = sys.argv[2:]

    if subcommand == "track":
        track(_TrackArgs().parse_args(sub_args))
    elif subcommand == "track_overlap":
        args = _OverLapTrackArgs().parse_args(sub_args)
        print(f"Testing model from {args.model_path} on {args.test_data}")
    else:
        print(f"Unknown subcommand: {subcommand}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # pragma: no cover
