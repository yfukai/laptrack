"""Command-line interface."""
from pathlib import Path
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import click
import geff
import pandas as pd
from tap import Tap

from . import data_conversion
from . import LapTrack


class LapTrackArgs(Tap):
    """Arguments for running LapTrack from the command line."""

    csv_path: Path  # path to input csv file
    output: Path = Path("output.geff")
    coordinate_cols: List[str] = ["position_x", "position_y"]
    frame_col: str = "frame"

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


def run(args: LapTrackArgs) -> None:
    """Execute tracking based on parsed arguments."""
    df = pd.read_csv(args.csv_path)

    lt_kwargs = {name: getattr(args, name) for name in LapTrack.model_fields}
    lt = LapTrack(**lt_kwargs)

    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=args.coordinate_cols, frame_col=args.frame_col
    )

    geff_tree = data_conversion.convert_dataframes_to_geff_networkx(
        track_df, split_df, merge_df, args.coordinate_cols, frame_col=args.frame_col
    )
    geff.write_nx(geff_tree, args.output)


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.version_option()
@click.pass_context
def main(ctx: click.Context) -> None:
    """LapTrack."""
    args = LapTrackArgs().parse_args(ctx.args)
    run(args)


if __name__ == "__main__":
    main(prog_name="laptrack")  # pragma: no cover
