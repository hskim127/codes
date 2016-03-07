#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
from scipy.sparse import coo_matrix
cimport numpy as np

ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE

def print_sparse(m):
    cdef np.ndarray[cINT32, ndim=1] row, col
    cdef np.ndarray[cDOUBLE, ndim=1] data
    cdef int i
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    row = m.row.astype(np.int32)
    col = m.col.astype(np.int32)
    data = m.data.astype(np.float64)
    for i in range(np.shape(data)[0]):
        print row[i], col[i], data[i]