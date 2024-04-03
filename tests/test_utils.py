import numpy as np

from laptrack.utils import _coord_is_empty


def test_coord_is_empty():
    assert _coord_is_empty(None)
    assert _coord_is_empty([])
    assert _coord_is_empty(np.array([]))
    assert not _coord_is_empty([1, 2])
