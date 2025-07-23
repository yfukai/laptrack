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


def _tap_configure_from_cls(clses: List[type]) -> Callable:
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

        for c in clses:
            for name, field in c.model_fields.items():
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


class _Args(Tap):
    """Base class for command-line arguments."""

    def configure(self) -> None:
        """Configure parser based on :class:`LapTrack` fields."""
        self.add_subparsers(dest="cmd", help="Subcommands for laptrack")
        self.add_subparser(
            "track", _TrackArgs, help="Track point-like object from a dataframe."
        )
        self.add_subparser("overlap_track", _OverLapTrackArgs, help="b help")


class _TrackArgs(Tap):
    csv_path: Optional[Path]  # path to input csv file
    track_geff_path: Optional[Path] = None  # path to output geff file
    output_path: Path
    metadata_path: Optional[Path] = None  # path to metadata file
    frame_col: str = "frame"
    coordinate_cols: List[str]
    configure = _tap_configure_from_cls([LapTrack])


class _OverLapTrackArgs(Tap):
    labels_path: Path  # path to input csv file
    metadata_path: Optional[Path] = None  # path to metadata file
    output_path: Path
    configure = _tap_configure_from_cls([OverLapTrack])


def track(args: _TrackArgs) -> None:
    """Execute tracking based on parsed arguments."""
    lt_kwargs = {name: getattr(args, name) for name in LapTrack.model_fields}
    lt = LapTrack(**lt_kwargs)

    if args.metadata_path is not None:
        with open(args.metadata_path, "r") as f:
            metadata = geff.GeffMetadata.model_validate_json(f.read())
    else:
        metadata = geff.GeffMetadata(
            geff_version=geff.__version__,
            directed=True,
        )

    if args.csv_path is None and args.track_geff_path is None:
        raise ValueError("Either csv_path or track_geff_path must be provided.")
    if args.csv_path is not None and args.track_geff_path is not None:
        raise ValueError("Only one of csv_path or track_geff_path can be provided.")
    if args.csv_path is not None:
        df = pd.read_csv(args.csv_path)
        track_df, split_df, merge_df = lt.predict_dataframe(
            df, coordinate_cols=args.coordinate_cols, frame_col=args.frame_col
        )
        geff_tree = data_conversion.dataframes_to_geff_networkx(
            track_df,
            split_df,
            merge_df,
            frame_col=args.frame_col,
        )
    else:
        tracked_tree, tracked_metadata = geff.read_nx(args.track_geff_path)
        tree, coords, mappings = data_conversion.geff_networkx_to_tree_coords_mapping(
            tracked_tree, coordinate_cols=args.coordinate_cols, frame_col=args.frame_col
        )
        rev_mappings = {v: k for k, v in mappings.items()}
        tree2 = lt.predict(coords, tree.edges, split_merge_validation=False)
        # Copy the GEFF to new output
        for edge in tree2.edges:
            edge2 = [rev_mappings[edge[0]], rev_mappings[edge[1]]]
            if edge2 not in tracked_tree.edges:
                assert edge2[0] not in tracked_tree.nodes
                assert edge2[1] not in tracked_tree.nodes
                tracked_tree.add_edge(edge2[0], edge2[1])
        _metadata = tracked_metadata.model_copy()
        _metadata.update(metadata)
        metadata = _metadata

    geff.write_nx(geff_tree, args.output_path, metadata=metadata)


def track_overlap(args: _OverLapTrackArgs) -> None:
    """Execute overlap tracking based on parsed arguments."""
    labels = _read_image(args.labels_path)
    tracked_tree = geff.read_nx(args.track_geff_path)

    lt_kwargs = {name: getattr(args, name) for name in OverLapTrack.model_fields}
    lt = OverLapTrack(**lt_kwargs)

    track_df, split_df, merge_df = lt.predict_overlap_dataframe(labels)

    geff_tree = data_conversion.dataframes_to_geff_networkx(
        track_df, split_df, merge_df, coordinate_cols=["seg_id"], frame_col="frame"
    )
    geff.write_nx(geff_tree, args.output_path)


def main():  # noqa: D103
    args = _Args().parse_args()
    # raise an error if no subcommand is provided
    if not hasattr(args, "cmd"):
        print("No subcommand provided. Use 'track' or 'overlap_track'.")
        sys.exit(1)
    if args.cmd == "track":
        track(args)
    del args.cmd  # Remove cmd from args to avoid confusion


if __name__ == "__main__":
    main()  # pragma: no cover
