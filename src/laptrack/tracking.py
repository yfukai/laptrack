from typing import List, Literal, Optional, Union

import numpy as np
from numpy import typing as npt

from ._typing_utils import FloatArray, Float

def track_points(coords : List[FloatArray], 
                 props : Optional[FloatArray] = None,
                 track_distance_cutoff : Optional[Float] = 15,
                 gap_closing_cutoff : Optional[Union[Float,Literal[False]]] = 15,
                 segment_splitting_cutoff : Optional[Union[Float,Literal[False]]] = False,
                 segment_merging_cutoff : Optional[Union[Float,Literal[False]]] = False,
                 ) -> List[npt.NDArray[np.uint32]]:
    pass