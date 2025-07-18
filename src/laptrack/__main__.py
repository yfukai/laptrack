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
import pandas as pd
from tap import Tap

from . import data_conversion
from . import LapTrack
from . import OverLapTrack

# %%


def __tap_configure_from_cls(cls: type) -> Callable:
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


class __TrackArgs(Tap):
    csv_path: Path  # path to input csv file
    output_path: Path
    frame_col: str = "frame"
    coordinate_cols: List[str]
    configure = __tap_configure_from_cls(LapTrack)


class __OverLapTrackArgs(Tap):
    labels_path: Path  # path to input csv file
    output_path: Path
    configure = __tap_configure_from_cls(OverLapTrack)


def track(args: __TrackArgs) -> None:
    """Execute tracking based on parsed arguments."""
    df = pd.read_csv(args.csv_path)

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


def track_overlap(args: __OverLapTrackArgs) -> None:
    """Execute overlap tracking based on parsed arguments."""
    labels = pd.read_csv(args.labels_path)

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
        track(__TrackArgs().parse_args(sub_args))
    elif subcommand == "track_overlap":
        args = __OverLapTrackArgs().parse_args(sub_args)
        print(f"Testing model from {args.model_path} on {args.test_data}")
    else:
        print(f"Unknown subcommand: {subcommand}")
        sys.exit(1)


if __name__ == "__main__":
    main()  # pragma: no cover

# %%
