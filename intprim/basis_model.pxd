cimport numpy as np
ctypedef np.float64_t DTYPE_t

cdef class GaussianBasisModel:
    cdef public np.int degree
    cdef public DTYPE_t scale
    cdef public DTYPE_t[:] centers

    cdef public np.ndarray[DTYPE_t, ndim = 2] get_basis_functions(self, np.ndarray[DTYPE_t, ndim = 1] x)
    cdef public np.ndarray[DTYPE_t, ndim = 2] get_basis_function_derivatives(self, np.ndarray[DTYPE_t, ndim = 1] x)
    cdef public np.ndarray[DTYPE_t, ndim = 2] get_block_diagonal_basis_matrix(self, np.ndarray[DTYPE_t, ndim = 1] x, np.int_t columns)
    cdef public np.ndarray[DTYPE_t, ndim = 2] get_block_diagonal_basis_matrix_derivative(self, np.ndarray[DTYPE_t, ndim = 1] x, np.int_t columns)
    cpdef public fit_basis_functions_linear_closed_form(self, np.ndarray[DTYPE_t, ndim = 1] x, np.ndarray[DTYPE_t, ndim = 2] y)
    cpdef public plot(self)
    cpdef public plot_weighted(self, coefficients, coefficient_names)
