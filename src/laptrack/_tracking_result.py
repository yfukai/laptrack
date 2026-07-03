"""Container for the tracking results."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import networkx as nx
import pandas as pd

from ._typing_utils import Int
from ._typing_utils import NumArray
from .data_conversion import digraph_to_geff_networkx
from .data_conversion import tree_to_dataframe


@dataclass
class TrackingResult:
    """The container of the tracking result.

    Parameters
    ----------
    tree : nx.DiGraph
        The directed graph for the tracks, whose nodes are `(frame, index)`.
        The edge direction represents the time order.
    coords : Optional[List[NumArray]], default None
        The list of coordinates of the points for each frame.
        The array index means `(sample, dimension)`.
    dataframe : Optional[pd.DataFrame], default None
        The original dataframe used for the tracking, if the tracking was
        performed with a dataframe input.
    frame_index : Optional[List[Tuple[int, int]]], default None
        The `(frame, index)` for each row (`iloc`) of `dataframe`.
        Required if `dataframe` is not None.
    coordinate_cols : Optional[List[str]], default None
        The list of the column names used for the coordinates.
    frame_col : str, default "frame"
        The column name used for the integer frame index.
    """

    tree: nx.DiGraph
    coords: Optional[List[NumArray]] = None
    dataframe: Optional[pd.DataFrame] = None
    frame_index: Optional[List[Tuple[int, int]]] = None
    coordinate_cols: Optional[List[str]] = None
    frame_col: str = "frame"

    def to_dataframes(
        self, index_offset: Int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert the result to the track, split and merge dataframes.

        Parameters
        ----------
        index_offset : Int, default 0
            The offset to add to the track and tree ids.

        Returns
        -------
        track_df : pd.DataFrame
            The track dataframe, with the following columns:

            - "frame" : The frame index.
            - "index" : The coordinate index.
            - "track_id" : The track id.
            - "tree_id" : The tree id.
            - the other columns : The coordinate values
              (or the columns of `dataframe` if it is not None).
        split_df : pd.DataFrame
            The splitting dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        merge_df : pd.DataFrame
            The merging dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        """
        if self.dataframe is not None:
            if self.frame_index is None:
                raise ValueError(
                    "frame_index must not be None if dataframe is not None."
                )
            track_df, split_df, merge_df = tree_to_dataframe(
                self.tree,
                dataframe=self.dataframe,
                frame_index=self.frame_index,
            )
        else:
            track_df, split_df, merge_df = tree_to_dataframe(
                self.tree,
                coords=self.coords,
            )

        track_df["track_id"] = track_df["track_id"] + index_offset
        track_df["tree_id"] = track_df["tree_id"] + index_offset
        for df in [split_df, merge_df]:
            if not df.empty:
                df["parent_track_id"] = df["parent_track_id"] + index_offset
                df["child_track_id"] = df["child_track_id"] + index_offset

        return track_df, split_df, merge_df

    def to_indices_networkx(self) -> nx.DiGraph:
        """Return the track tree whose nodes are `(frame, index)` tuples.

        Returns
        -------
        tree : nx.DiGraph
            The copy of the track tree. The edge direction represents
            the time order.
        """
        return self.tree.copy()

    def to_geff_networkx(self, attr_names: Optional[List[str]] = None) -> nx.DiGraph:
        """Convert the result to a networkx graph in the GEFF format.

        Parameters
        ----------
        attr_names : Optional[List[str]], default None
            The list of the node attribute names for the frame and coordinates.
            If None, `[frame_col] + coordinate_cols` is used if
            `coordinate_cols` is not None, otherwise the default names
            ("frame", "coord-0", "coord-1", ...) are used.

        Returns
        -------
        geff_tree : nx.DiGraph
            The directed graph in the GEFF format, with integer node labels
            and the frame and coordinate node attributes.
        """
        if (
            attr_names is None
            and self.coords is not None
            and self.coordinate_cols is not None
        ):
            attr_names = [self.frame_col, *self.coordinate_cols]
        return digraph_to_geff_networkx(self.tree, self.coords, attr_names=attr_names)

    def write_csvs(
        self,
        directory: Union[str, PathLike],
        prefix: str = "",
        index_offset: Int = 0,
    ) -> Dict[str, Path]:
        """Write the track, split and merge dataframes as CSV files.

        Parameters
        ----------
        directory : Union[str, PathLike]
            The directory to write the CSV files to. Created if not existing.
        prefix : str, default ""
            The prefix prepended to the file names
            ("track.csv", "split.csv", "merge.csv").
        index_offset : Int, default 0
            The offset to add to the track and tree ids.

        Returns
        -------
        paths : Dict[str, Path]
            The paths of the written files, with the keys
            "track", "split" and "merge".
        """
        track_df, split_df, merge_df = self.to_dataframes(index_offset=index_offset)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, df in [
            ("track", track_df),
            ("split", split_df),
            ("merge", merge_df),
        ]:
            path = directory / f"{prefix}{name}.csv"
            write_index = any(n is not None for n in df.index.names)
            df.to_csv(path, index=write_index)
            paths[name] = path
        return paths

    def write_geff(
        self,
        path: Union[str, PathLike],
        metadata: Optional[Any] = None,
        attr_names: Optional[List[str]] = None,
        **write_kwargs: Any,
    ) -> None:
        """Write the result to a GEFF (Graph Exchange File Format) zarr store.

        Parameters
        ----------
        path : Union[str, PathLike]
            The path of the zarr store to write to.
        metadata : Optional[geff.GeffMetadata], default None
            The GEFF metadata. If None, a minimal metadata is generated.
        attr_names : Optional[List[str]], default None
            The node attribute names for the frame and coordinates.
            See `to_geff_networkx`.
        **write_kwargs : Any
            The additional keyword arguments passed to `geff.write`.

        Raises
        ------
        ImportError
            If the `geff` package is not installed.
        """
        try:
            import geff
        except ImportError:
            raise ImportError(
                "Please install `geff` to use `write_geff` "
                "(for example, by `pip install laptrack[geff]`)."
            )
        geff_tree = self.to_geff_networkx(attr_names=attr_names)
        geff.write(geff_tree, path, metadata=metadata, **write_kwargs)


def _connected_edges_to_frame_index_edges(
    connected_edges: Optional[Sequence[Tuple[Int, Int]]],
    frame_index: Sequence[Tuple[int, int]],
) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Convert the connected edges in ilocs to the (frame, index) format."""
    if connected_edges is None:
        return None
    return [(frame_index[int(i1)], frame_index[int(i2)]) for i1, i2 in connected_edges]
