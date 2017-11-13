#!python

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import scipy.linalg
import scipy.optimize

cimport numpy as np

DTYPE = np.float64

cdef class GaussianBasisModel(object):
    def __init__(self, degree):
        self.degree = degree
        self.scale = 0.01
        self.centers = np.linspace(0, 1, self.degree, dtype = DTYPE)

    cdef public np.ndarray[DTYPE_t, ndim = 2] get_basis_functions(self, np.ndarray[DTYPE_t, ndim = 1] x):
        f = lambda x, c: np.exp(-(np.array([x - y for y in c], dtype = DTYPE) ** 2) / (2.0 * self.scale))

        return f(x, self.centers)

    cdef public np.ndarray[DTYPE_t, ndim = 2] get_basis_function_derivatives(self, np.ndarray[DTYPE_t, ndim = 1] x):
        f = lambda x, c: (np.exp(-(np.array([y - x for y in c], dtype = DTYPE) ** 2) / (2.0 * self.scale)) * np.array([y - x for y in c], dtype = DTYPE)) / self.scale

        return f(x, self.centers)

    cdef public np.ndarray[DTYPE_t, ndim = 2] get_block_diagonal_basis_matrix(self, np.ndarray[DTYPE_t, ndim = 1] x, np.int_t columns):
        return scipy.linalg.block_diag(*np.tile(self.get_basis_functions(x), (1, columns)).T).T

    cdef public np.ndarray[DTYPE_t, ndim = 2] get_block_diagonal_basis_matrix_derivative(self, np.ndarray[DTYPE_t, ndim = 1] x, np.int_t columns):
        # Matrix will be of dimension num_dof * len(x) x (num_dof * basis_degree) * len(x)
        return scipy.linalg.block_diag(*np.tile(self.get_basis_function_derivatives(x), (1, columns)).T).T

    # Closed form solution for linear least squares problem
    cpdef public fit_basis_functions_linear_closed_form(self, np.ndarray[DTYPE_t, ndim = 1] x, np.ndarray[DTYPE_t, ndim = 2] y):
        cdef np.ndarray[DTYPE_t, ndim = 2] basis_matrix = self.get_basis_functions(x).T

        return np.linalg.solve(np.dot(basis_matrix.T, basis_matrix), np.dot(basis_matrix.T, y))

    cpdef public plot(self):
        cdef np.ndarray[DTYPE_t, ndim = 1] test_domain = np.linspace(0, 1, 100, dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] test_range = self.get_basis_functions(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Functions')

        plt.show(block = False)

    cpdef public plot_weighted(self, coefficients, coefficient_names):
        cdef np.ndarray[DTYPE_t, ndim = 1] test_domain = np.linspace(0, 1, 100, dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] test_range = self.get_basis_functions(test_domain)

        for coefficients_dimension, name in zip(coefficients, coefficient_names):
            fig = plt.figure()

            for basis_func, coefficient in zip(test_range, coefficients_dimension):
                plt.plot(test_domain, basis_func * coefficient)

            fig.suptitle('Basis Functions For Dimension ' + name)

        plt.show(block = False)
