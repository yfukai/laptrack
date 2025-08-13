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
from pydantic import BaseModel
from skimage.io import imread
from tap import Tap

from . import data_conversion
from . import LapTrack
from . import OverLapTrack


def _read_image(file_path):
    extension = Path(file_path).suffix.lower()
    if any(ext in str(file_path) for ext in [".zarr", ".zarr.zip", ".zr", ".zr.zip"]):
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


def _tap_configure_from_cls(clses: List[type[BaseModel]]) -> Callable:
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
            "track", TrackArgs, help="Track point-like object from a dataframe."
        )
        self.add_subparser(
            "overlap_track",
            OverLapTrackArgs,
            help="Track segmentation images by overlap.",
        )


class TrackArgs(Tap):
    """Arguments for the "track" command."""

    csv_path: Optional[Path] = None  # path to input csv file
    track_geff_path: Optional[Path] = None  # path to output geff file
    output_path: Path
    metadata_path: Optional[Path] = None  # path to metadata file
    frame_col: str = "frame"
    coordinate_cols: List[str]
    configure = _tap_configure_from_cls([LapTrack])


class OverLapTrackArgs(Tap):
    """Arguments for the "overlap_track" command."""

    labels_path: Path  # path to input csv file
    image_path: Path | None = None  # path to input csv file
    metadata_path: Optional[Path] = None  # path to metadata file
    output_path: Path
    append_relative_objects: bool = False
    configure = _tap_configure_from_cls([OverLapTrack])


def track(args: TrackArgs) -> None:
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
        geff_nxs = data_conversion.dataframes_to_geff_networkx(
            track_df,
            split_df,
            merge_df,
            frame_col=args.frame_col,
        )
    else:
        geff_nx, tracked_metadata = geff.read_nx(args.track_geff_path)
        tree, coords, mappings = data_conversion.geff_networkx_to_tree_coords_mapping(
            geff_nx,
            coordinate_props=args.coordinate_cols,
            frame_prop=args.frame_col,
        )
        rev_mappings = {v: k for k, v in mappings.items()}
        tree2 = lt.predict(coords, tree.edges, split_merge_validation=False)
        # Copy the GEFF to new output
        for edge in tree2.edges:
            edge2 = [rev_mappings[edge[0]], rev_mappings[edge[1]]]
            if edge2 not in geff_nx.edges:
                assert edge2[0] in geff_nx.nodes
                assert edge2[1] in geff_nx.nodes
                geff_nx.add_edge(edge2[0], edge2[1])
        _metadata = tracked_metadata.model_copy(
            update=metadata.model_dump(exclude_unset=True)
        )
        metadata = _metadata
        geff_nxs = data_conversion.GEFFNetworkXs(tree=geff_nx)

    geff.write_nx(geff_nxs.tree, args.output_path, metadata=metadata)
    geff.write_nx(geff_nxs.lineage_tree, args.output_path / "lineage_tree.geff")


def overlap_track(args: OverLapTrackArgs) -> None:
    """Execute overlap tracking based on parsed arguments."""
    labels = _read_image(args.labels_path)

    lt_kwargs = {name: getattr(args, name) for name in OverLapTrack.model_fields}
    lt = OverLapTrack(**lt_kwargs)

    if args.metadata_path is not None:
        with open(args.metadata_path, "r") as f:
            metadata = geff.GeffMetadata.model_validate_json(f.read())
    else:
        metadata = geff.GeffMetadata(
            geff_version=geff.__version__,
            directed=True,
        )
    if args.append_relative_objects:
        related_obj_path = args.labels_path.absolute().relative_to(
            args.output_path.absolute()
        )
        robjs = [] if metadata.related_objects is None else metadata.related_objects
        metadata.related_objects = [
            *robjs,
            geff.metadata_schema.RelatedObject(
                path=related_obj_path, type="labels", label_prop=["seg_id"]
            ),
        ]
        if args.image_path is not None:
            related_obj_path = args.image_path.absolute().relative_to(
                args.output_path.absolute()
            )
            metadata.related_objects.append(
                geff.metadata_schema.RelatedObject(
                    path=related_obj_path,
                    type="image",
                )
            )

    track_df, split_df, merge_df = lt.predict_overlap_dataframe(labels)
    track_df = track_df.reset_index(drop=False)
    geff_nxs = data_conversion.dataframes_to_geff_networkx(
        track_df[["frame", "label", "track_id"]], split_df, merge_df, frame_col="frame"
    )
    geff.write_nx(geff_nxs.tree, args.output_path, metadata=metadata)
    geff.write_nx(geff_nxs.lineage_tree, args.output_path / "lineage_tree.geff")


def main():  # noqa: D103
    args = _Args().parse_args()
    # raise an error if no subcommand is provided
    if not hasattr(args, "cmd"):
        print("No subcommand provided. Use 'track' or 'overlap_track'.")
        sys.exit(1)
    if args.cmd == "track":
        track(args)
    elif args.cmd == "overlap_track":
        overlap_track(args)


if __name__ == "__main__":
    main()  # pragma: no cover
