"""Solver for `tracksdata <https://github.com/royerlab/tracksdata>`_ graphs using LapTrack."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

try:
    import polars as pl
    from tracksdata.attrs import EdgeAttr
    from tracksdata.attrs import NodeAttr
    from tracksdata.constants import DEFAULT_ATTR_KEYS
    from tracksdata.graph._base_graph import BaseGraph
    from tracksdata.graph._graph_view import GraphView
    from tracksdata.solvers._base_solver import BaseSolver
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Please install `tracksdata` to use `laptrack.tracksdata_solver` "
        "(for example, by `pip install tracksdata`)."
    ) from e

from ._tracking import LapTrack
from ._typing_utils import NumArray


class LapTrackSolver(BaseSolver):
    """Solve a tracksdata tracking problem with LapTrack.

    The solver reads the frame and coordinate attributes from the graph
    nodes, executes `LapTrack.predict`, and marks the nodes and edges of
    the resulting tracks with `output_key`. The edges predicted by LapTrack
    that do not exist in the graph are added, so the graph does not need
    candidate edges beforehand.

    Parameters
    ----------
    tracker : Optional[LapTrack]
        The LapTrack instance defining the tracking parameters.
        If None, a LapTrack instance with the default parameters is used.
    coordinate_attr_keys : Optional[Sequence[str]]
        The node attribute keys for the coordinates. If None, the existing
        keys among ("z", "y", "x") are used in this order.
    frame_attr_key : str
        The node attribute key for the frame index.
    output_key : str
        The node/edge attribute key to store the solution boolean values.
    reset : bool
        Whether to reset the existing solution values before solving.
    return_solution : bool
        Whether to return the solution graph view.
    """

    def __init__(
        self,
        tracker: Optional[LapTrack] = None,
        coordinate_attr_keys: Optional[Sequence[str]] = None,
        frame_attr_key: str = DEFAULT_ATTR_KEYS.T,
        output_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
        reset: bool = True,
        return_solution: bool = True,
    ):
        super().__init__(
            output_key=output_key,
            reset=reset,
            return_solution=return_solution,
        )
        self.tracker = tracker if tracker is not None else LapTrack()
        self.coordinate_attr_keys = coordinate_attr_keys
        self.frame_attr_key = frame_attr_key

    def _get_coordinate_attr_keys(self, graph: "BaseGraph") -> List[str]:
        if self.coordinate_attr_keys is not None:
            return list(self.coordinate_attr_keys)
        default_keys = [
            DEFAULT_ATTR_KEYS.Z,
            DEFAULT_ATTR_KEYS.Y,
            DEFAULT_ATTR_KEYS.X,
        ]
        node_attr_keys = graph.node_attr_keys()
        coordinate_attr_keys = [key for key in default_keys if key in node_attr_keys]
        if not coordinate_attr_keys:
            raise ValueError(
                "No coordinate attribute keys found in the graph. "
                "Please specify `coordinate_attr_keys` explicitly."
            )
        return coordinate_attr_keys

    def _graph_to_coords(
        self, graph: "BaseGraph", coordinate_attr_keys: List[str]
    ) -> Tuple[List[NumArray], Dict[Tuple[int, int], int]]:
        """Convert the graph nodes to the coordinate list and the node id map."""
        nodes_df = graph.node_attrs(
            attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, self.frame_attr_key]
            + list(coordinate_attr_keys)
        )
        frames = nodes_df[self.frame_attr_key].to_numpy()
        node_ids = nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_numpy()
        coord_values = nodes_df[list(coordinate_attr_keys)].to_numpy()

        frame_min = int(frames.min())
        frame_max = int(frames.max())
        coords: List[NumArray] = []
        frame_index_to_node_id: Dict[Tuple[int, int], int] = {}
        for frame in range(frame_min, frame_max + 1):
            indices = np.where(frames == frame)[0]
            coords.append(coord_values[indices])
            for index, node_index in enumerate(indices):
                frame_index_to_node_id[(frame - frame_min, index)] = int(
                    node_ids[node_index]
                )
        return coords, frame_index_to_node_id

    def solve(self, graph: "BaseGraph") -> Optional["GraphView"]:
        """Solve the tracking problem with LapTrack.

        Parameters
        ----------
        graph : BaseGraph
            The graph with the frame and coordinate node attributes. The
            graph is modified in-place: the solution attributes are added to
            the nodes and edges, and the predicted edges missing in the graph
            are added.

        Returns
        -------
        GraphView | None
            The graph view of the solution if `return_solution` is True,
            otherwise None.
        """
        if graph.num_nodes() == 0:
            raise ValueError("No nodes found in the graph.")

        coordinate_attr_keys = self._get_coordinate_attr_keys(graph)
        coords, frame_index_to_node_id = self._graph_to_coords(
            graph, coordinate_attr_keys
        )

        tree = self.tracker.predict(coords)
        solution_node_pairs = [
            (frame_index_to_node_id[node1], frame_index_to_node_id[node2])
            for node1, node2 in tree.edges()
        ]

        # prepare the solution attribute keys
        if self.output_key not in graph.edge_attr_keys():
            graph.add_edge_attr_key(self.output_key, pl.Boolean, default_value=False)
        elif self.reset:
            graph.update_edge_attrs(attrs={self.output_key: False})
        if self.output_key not in graph.node_attr_keys():
            graph.add_node_attr_key(self.output_key, pl.Boolean, default_value=False)
        elif self.reset:
            graph.update_node_attrs(attrs={self.output_key: False})

        # mark the existing edges as the solution, and add the missing ones
        edges_df = graph.edge_attrs(attr_keys=None)
        edge_id_map = {
            (int(source), int(target)): int(edge_id)
            for edge_id, source, target in zip(
                edges_df[DEFAULT_ATTR_KEYS.EDGE_ID].to_numpy(),
                edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
                edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
            )
        }
        existing_edge_ids = []
        for source, target in solution_node_pairs:
            edge_id = edge_id_map.get((source, target))
            if edge_id is not None:
                existing_edge_ids.append(edge_id)
            else:
                graph.add_edge(
                    source, target, {self.output_key: True}, validate_keys=False
                )
        if existing_edge_ids:
            graph.update_edge_attrs(
                edge_ids=existing_edge_ids, attrs={self.output_key: True}
            )

        # mark the solution nodes
        solution_node_ids = np.unique(
            np.array(solution_node_pairs, dtype=np.int64).ravel()
        )
        if len(solution_node_ids) > 0:
            graph.update_node_attrs(
                node_ids=solution_node_ids, attrs={self.output_key: True}
            )

        if self.return_solution:
            return graph.filter(
                NodeAttr(self.output_key) == True,  # noqa: E712
                EdgeAttr(self.output_key) == True,  # noqa: E712
            ).subgraph()
        return None
