from collections.abc import Sized
from typing import cast
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from ._typing_utils import Float
from ._typing_utils import Int
from ._typing_utils import IntArray
from ._typing_utils import Matrix
from ._typing_utils import NumArray


IndexType = Union[Sequence[Int], IntArray]
DataType = Union[Sequence[Union[Int, Float]], NumArray]


class coo_matrix_builder:  # noqa: N801
    """store data to build scipy.sparce.coo_matrix."""

    def __init__(
        self,
        shape: Tuple[Int, Int],
        row: Optional[IndexType] = None,
        col: Optional[IndexType] = None,
        data: Optional[DataType] = None,
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
        row: Union[Int, IndexType],
        col: Union[Int, IndexType],
        data: Union[Int, Float, DataType],
    ) -> None:
        """Append data from row, column index and data values.

        Parameters
        ----------
        row : Union[Int, IndexType]
            The row values.
        col : Union[Int, IndexType]
            The column values.
        data : Union[Int, Float, DataType]
            The data values.
        """
        if isinstance(row, Sized) or isinstance(col, Sized) or isinstance(data, Sized):
            count = None
            if isinstance(row, Sized):
                # dirty hack but will be solved in Python 3.10
                # https://stackoverflow.com/questions/65912706/pattern-matching-over-nested-union-types-in-python # noqa :
                row2 = np.array(cast(IndexType, row))
                count = len(row2)
                if isinstance(col, Sized):
                    col2 = np.array(cast(IndexType, col))
                    assert len(col2) == count
                else:
                    col2 = np.ones(count, dtype=self.index_dtype) * col
            else:
                assert isinstance(col, Sized)
                col2 = np.array(cast(IndexType, col))
                count = len(col2)
                row2 = np.ones(count, dtype=self.index_dtype) * row

            if isinstance(data, Sized):
                data2 = np.array(cast(DataType, data))
                assert len(data2) == count
            else:
                data2 = np.ones(count, dtype=self.dtype) * data
        else:
            row2 = np.array([cast(Int, row)])
            col2 = np.array([cast(Int, col)])
            data2 = np.array([cast(Union[Int, Float], data)])
            count = 1

        if count == 0:
            return
        assert len(row2) == count
        assert len(col2) == count
        assert len(data2) == count
        self.row = np.concatenate([self.row, row2], dtype=self.index_dtype)
        self.col = np.concatenate([self.col, col2], dtype=self.index_dtype)
        self.data = np.concatenate([self.data, data2], dtype=self.dtype)

    def append_matrix(
        self,
        matrix: Union["coo_matrix_builder", Matrix],
        shift: Tuple[Int, Int] = (0, 0),
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
            matrix2 = coo_matrix(matrix)
        else:
            matrix2 = matrix
        self.append(matrix2.row + shift[0], matrix2.col + shift[1], matrix2.data)

    def to_coo_matrix(self) -> coo_matrix:
        """Generate `coo_matrix`.

        Returns
        -------
        matrix : coo_matrix
            the generated coo_matrix
        """
        return coo_matrix(
            (self.data, (self.row, self.col)),
            shape=(self.n_row, self.n_col),
        )

    def __setitem__(
        self, index: Tuple[Union[Int, IndexType], Union[Int, IndexType]], value
    ):
        """Syntax sugar for append.

        Parameters
        ----------
        index : Union[Tuple[Int], Tuple[Sequence[Int]]]
            The index to append values. The length must be 2 (row,col).
        value : [type]
            The data values to append.
        """
        assert len(index) == 2
        self.append(index[0], index[1], value)

    @property
    def T(self) -> "coo_matrix_builder":  # noqa : N802
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
