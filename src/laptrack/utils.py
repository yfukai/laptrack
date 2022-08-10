"""Miscellous utilities."""
from typing import List
from typing import Tuple

from ._typing_utils import EdgeType
from ._typing_utils import Int


def order_edges(edges: EdgeType) -> List[Tuple[Tuple[Int, Int], Tuple[Int, Int]]]:
    """
    Order edges so that it points to the temporal order.

    Parameters
    ----------
    edges : EdgeType
        the list of edges. assumes ((frame1,index1), (frame2,index2)) for each edge

    Returns
    -------
    edges : List[Tuple[Tuple[Int,Int],Tuple[Int,Int]]]
        the sorted edges

    """
    return [(n1, n2) if n1[0] < n2[0] else (n2, n1) for (n1, n2) in edges]
