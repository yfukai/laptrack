from collections.abc import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from ._typing_utils import Float
from ._typing_utils import Int
from ._typing_utils import Matrix
from ._typing_utils import NumArray


class coo_matrix_builder:  # noqa: N801,D101 as scipy uses the same convention

    ...


class coo_matrix_builder:  # noqa: N801
    """store data to build scipy.sparce.coo_matrix."""

    def __init__(
        self,
        shape: Tuple[Int, Int],
        row: Optional[Sequence[Int]] = None,
        col: Optional[Sequence[Int]] = None,
        data: Optional[Sequence[Union[Int, Float]]] = None,
        *,
        dtype: npt.DTypeLike = np.float64,
        index_dtype: npt.DTypeLike = np.int64,
    ) -> None:
        """Initialize the object.

        Parameters
        ----------
        n_row : Int
            The row count.
        n_col : Int
            The column count.
        row : Optional[Sequence[Int]], optional
            The row values. If None, an empty array will be created.
        col : Optional[Sequence[Int]], optional
            The column values. If None, an empty array will be created.
        data : Optional[Sequence[Union[Int, Float]]], optional
            The data values. If None, an empty array will be created.
        dtype : npt.DTypeLike, optional
            The dtype for the values, by default np.float64
        index_dtype : npt.DTypeLike, optional
            The dtype for the index, by default np.int64
        """
        assert len(shape) == 2
        if row is None:
            row = []
        if col is None:
            col = []
        if data is None:
            data = []
        self.n_row = shape[0]
        self.n_col = shape[1]
        assert isinstance(row, Iterable)
        assert isinstance(col, Iterable)
        assert isinstance(data, Iterable)
        assert len(row) == len(col)
        assert len(row) == len(data)
        self.row = np.array(row, dtype=index_dtype)
        self.col = np.array(col, dtype=index_dtype)
        self.data = np.array(data, dtype=dtype)
        self.shape = (self.n_row, self.n_col)
        self.dtype = dtype
        self.index_dtype = index_dtype

    def append(
        self,
        row: Union[Int, Sequence[Int]],
        col: Union[Int, Sequence[Int]],
        data: Union[NumArray, Int, Float],
    ) -> None:
        """Append data from row, column index and data values.

        Parameters
        ----------
        row : Union[Int, Sequence[Int]]
            The row values.
        col : Union[Int, Sequence[Int]]
            The column values.
        data : Union[NumArray, Int, Float]
            The data values.
        """
        if any([isinstance(val, Iterable) for val in [row, col, data]]):
            count = None
            if isinstance(row, Iterable):
                count = len(row)
                if isinstance(col, Iterable):
                    assert len(col) == count
                else:
                    col = np.ones(count, dtype=self.index_dtype) * col
            else:
                assert isinstance(col, Iterable)
                count = len(col)
                row = np.ones(count, dtype=self.index_dtype) * row

            if isinstance(data, Iterable):
                assert len(data) == count
            else:
                data = np.ones(count, dtype=self.dtype) * data
        else:
            row = [row]
            col = [col]
            data = [data]
            count = 1
        if count == 0:
            return
        assert len(row) == count
        assert len(col) == count
        assert len(data) == count
        self.row = np.concatenate([self.row, row], dtype=self.index_dtype)
        self.col = np.concatenate([self.col, col], dtype=self.index_dtype)
        self.data = np.concatenate([self.data, data], dtype=self.dtype)

    def append_matrix(
        self, matrix: Union[coo_matrix_builder, Matrix], shift: Tuple[Int, Int] = (0, 0)
    ) -> None:
        """Append data from another coo_matrix_builder.

        Parameters
        ----------
        matrix : coo_matrix_builder or Matrix
            the matrix to append
        shift : Tuple[Int,Int]
            the shift for row and column
        """
        assert len(shift)
        if not isinstance(matrix, coo_matrix_builder):
            matrix = coo_matrix(matrix)
        self.append(matrix.row + shift[0], matrix.col + shift[1], matrix.data)

    def to_coo_matrix(self) -> coo_matrix:
        """Generate `coo_matrix`.

        Returns
        -------
        matrix : coo_matrix
            the generated coo_matrix
        """
        return coo_matrix(
            (self.data, (self.row, self.col)), shape=(self.n_row, self.n_col)
        )

    def __setitem__(self, index: Union[Tuple[Int], Tuple[Sequence[Int]]], value):
        """Syntax sugar for append.

        Parameters
        ----------
        index : Union[Tuple[Int], Tuple[Sequence[Int]]]
            The index to append values. The length must be 2 (row,col).
        value : [type]
            The data values to append.
        """
        assert len(index) == 2
        self.append(*index, value)

    @property
    def T(self) -> coo_matrix_builder:  # noqa : N802
        """Transpose matrix.

        Returns
        -------
        builder : coo_matrix_builder
            the transposed builder
        """
        return coo_matrix_builder(self.shape[::-1], self.col, self.row, self.data)

    def size(self):
        """Get current array size.

        Returns
        -------
        size : int
            the array size for data
        """
        assert len(self.row) == len(self.col)
        assert len(self.row) == len(self.data)
        return len(self.data)
