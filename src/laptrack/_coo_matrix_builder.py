from collections.abc import Sized
from typing import cast
from typing import List
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
    """Buffered builder for ``scipy.sparse.coo_matrix``.

    Append-only triplets (row, col, data) are stored as a list of array
    chunks and concatenated lazily on read. This makes ``append`` O(1)
    amortized instead of O(N) per call (the prior implementation
    re-concatenated the full buffer on every append, leading to O(N²)
    cost when callers do many small appends in a loop).
    """

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
        shape : Tuple[Int, Int]
            The (row_count, col_count) of the matrix.
        row : Optional[Sequence[Int]], optional
            The initial row values. If None, an empty buffer.
        col : Optional[Sequence[Int]], optional
            The initial col values. If None, an empty buffer.
        data : Optional[Sequence[Union[Int, Float]]], optional
            The initial data values. If None, an empty buffer.
        dtype : npt.DTypeLike, optional
            The dtype for the values, by default np.float64.
        index_dtype : npt.DTypeLike, optional
            The dtype for the indices, by default np.int64.
        """
        assert len(shape) == 2
        if row is None:
            row = []
        if col is None:
            col = []
        if data is None:
            data = []
        assert len(row) == len(col)
        assert len(row) == len(data)
        self.n_row = shape[0]
        self.n_col = shape[1]
        self.shape = (self.n_row, self.n_col)
        self.dtype = dtype
        self.index_dtype = index_dtype
        self._row_chunks: List[np.ndarray] = []
        self._col_chunks: List[np.ndarray] = []
        self._data_chunks: List[np.ndarray] = []
        if len(row) > 0:
            self._row_chunks.append(np.asarray(row, dtype=index_dtype))
            self._col_chunks.append(np.asarray(col, dtype=index_dtype))
            self._data_chunks.append(np.asarray(data, dtype=dtype))

    @staticmethod
    def _flush(chunks: List[np.ndarray], dtype: npt.DTypeLike) -> np.ndarray:
        """Collapse ``chunks`` to a single array and return it."""
        if len(chunks) == 0:
            arr = np.empty(0, dtype=dtype)
            chunks.append(arr)
            return arr
        if len(chunks) == 1:
            return chunks[0]
        arr = np.concatenate(chunks, dtype=dtype)
        chunks.clear()
        chunks.append(arr)
        return arr

    @property
    def row(self) -> np.ndarray:
        """Materialized row indices."""
        return self._flush(self._row_chunks, self.index_dtype)

    @row.setter
    def row(self, value: IndexType) -> None:
        self._row_chunks = [np.asarray(value, dtype=self.index_dtype)]

    @property
    def col(self) -> np.ndarray:
        """Materialized column indices."""
        return self._flush(self._col_chunks, self.index_dtype)

    @col.setter
    def col(self, value: IndexType) -> None:
        self._col_chunks = [np.asarray(value, dtype=self.index_dtype)]

    @property
    def data(self) -> np.ndarray:
        """Materialized data values."""
        return self._flush(self._data_chunks, self.dtype)

    @data.setter
    def data(self, value: DataType) -> None:
        self._data_chunks = [np.asarray(value, dtype=self.dtype)]

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
                row2 = np.asarray(cast(IndexType, row), dtype=self.index_dtype)
                count = len(row2)
                if isinstance(col, Sized):
                    col2 = np.asarray(cast(IndexType, col), dtype=self.index_dtype)
                    assert len(col2) == count
                else:
                    col2 = np.full(count, col, dtype=self.index_dtype)
            else:
                assert isinstance(col, Sized)
                col2 = np.asarray(cast(IndexType, col), dtype=self.index_dtype)
                count = len(col2)
                row2 = np.full(count, row, dtype=self.index_dtype)

            if isinstance(data, Sized):
                data2 = np.asarray(cast(DataType, data), dtype=self.dtype)
                assert len(data2) == count
            else:
                data2 = np.full(count, data, dtype=self.dtype)
        else:
            row2 = np.array([cast(Int, row)], dtype=self.index_dtype)
            col2 = np.array([cast(Int, col)], dtype=self.index_dtype)
            data2 = np.array([cast(Union[Int, Float], data)], dtype=self.dtype)
            count = 1

        if count == 0:
            return
        assert len(row2) == count
        assert len(col2) == count
        assert len(data2) == count
        self._row_chunks.append(row2)
        self._col_chunks.append(col2)
        self._data_chunks.append(data2)

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
        assert len(shift) == 2
        if isinstance(matrix, coo_matrix_builder):
            # avoid forcing a flush of the source: walk its chunks directly
            for r, c, d in zip(
                matrix._row_chunks, matrix._col_chunks, matrix._data_chunks
            ):
                if len(r) == 0:
                    continue
                self.append(r + shift[0], c + shift[1], d)
        else:
            m = coo_matrix(matrix)
            self.append(m.row + shift[0], m.col + shift[1], m.data)

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
        out = coo_matrix_builder(
            self.shape[::-1], dtype=self.dtype, index_dtype=self.index_dtype
        )
        # share chunk references — read-only access on the new builder is
        # safe; further appends/setters on either side won't aliasing-corrupt
        # the other because they replace the *list*, not its contents.
        out._row_chunks = list(self._col_chunks)
        out._col_chunks = list(self._row_chunks)
        out._data_chunks = list(self._data_chunks)
        return out

    def size(self) -> int:
        """Get current array size.

        Returns
        -------
        size : int
            the array size for data
        """
        n = sum(len(c) for c in self._data_chunks)
        assert n == sum(len(c) for c in self._row_chunks)
        assert n == sum(len(c) for c in self._col_chunks)
        return n
