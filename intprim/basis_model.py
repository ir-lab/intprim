#!python
"""@package intprim
This module implements a simple linear basis model.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import scipy.linalg
import scipy.optimize

DTYPE = np.float64

class GaussianBasisModel(object):
    """The GaussianBasisModel class fits a linear Gaussian basis model to trajectories.
    """
    def __init__(self, degree):
        """The constructor function.
        """
        self.degree = degree
        self.scale = 0.01
        self.centers = np.linspace(0, 1, self.degree, dtype = DTYPE)

    def get_basis_functions(self, x):
        """Gets the basis functions for phase x.
        """
        f = lambda x, c: np.exp(-(np.array([x - y for y in c], dtype = DTYPE) ** 2) / (2.0 * self.scale))

        return f(x, self.centers)

    def get_basis_function_derivatives(self, x):
        """Gets the basis function derivatives for phase x.
        """
        f = lambda x, c: (np.exp(-(np.array([y - x for y in c], dtype = DTYPE) ** 2) / (2.0 * self.scale)) * np.array([y - x for y in c], dtype = DTYPE)) / self.scale

        return f(x, self.centers)

    def get_block_diagonal_basis_matrix(self, x, columns):
        """Gets the block diagonal basis matrix for phase x.
        """
        return scipy.linalg.block_diag(*np.tile(self.get_basis_functions(x), (1, columns)).T).T

    def get_block_diagonal_basis_matrix_derivative(self, x, columns):
        """Gets the block diagonal basis matrix derivative for phase x.
        """
        # Matrix will be of dimension num_dof * len(x) x (num_dof * basis_degree) * len(x)
        return scipy.linalg.block_diag(*np.tile(self.get_basis_function_derivatives(x), (1, columns)).T).T

    # Closed form solution for linear least squares problem
    def fit_basis_functions_linear_closed_form(self, x, y):
        """Calculates the weights for a given trajectory y given the phase values x.
        """
        basis_matrix = self.get_basis_functions(x).T

        return np.linalg.solve(np.dot(basis_matrix.T, basis_matrix), np.dot(basis_matrix.T, y))

    def plot(self):
        """Plots the unweighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype = DTYPE)
        test_range = self.get_basis_functions(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Functions')

        plt.show(block = False)

    def plot_weighted(self, coefficients, coefficient_names):
        """Plots the weighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype = DTYPE)
        test_range = self.get_basis_functions(test_domain)

        for coefficients_dimension, name in zip(coefficients, coefficient_names):
            fig = plt.figure()

            for basis_func, coefficient in zip(test_range, coefficients_dimension):
                plt.plot(test_domain, basis_func * coefficient)

            fig.suptitle('Basis Functions For Dimension ' + name)

        plt.show(block = False)
