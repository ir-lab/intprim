#!python
"""@package intprim
This module implements Bayesian Interaction Primitives and the corresponding EKF SLAM algorithm.
"""

import basis_model
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pickle
import scipy.linalg

DTYPE = np.float64
DEFAULT_NUM_SAMPLES = 100

class EKFSLAM(object):
    """The EKFSLAM class localizes an interaction in time and space via EKF SLAM.
    """
    def __init__(self, mean_basis_weights, cov_basis_weights, basis_model, measurement_dimension, phase_velocity, phase_var):
        """The constructor function.
        """
        self.state_dimension = 2 + len(mean_basis_weights)

        self.measurement_dimension = measurement_dimension

        # Initial phase is 0 while the landmarks are the mean basis weights of the demonstrations.
        self.state_mean = np.zeros(self.state_dimension, dtype = DTYPE)
        self.state_mean[0] = 0.0
        self.state_mean[1] = phase_velocity
        self.state_mean[2:] = mean_basis_weights.astype(DTYPE)

        # Covariance starts at 0 for phase since we assume trajectories start at the initial point
        # The covariance for the basis weights is the same as computed from demonstrations
        self.state_cov = np.zeros((self.state_dimension, self.state_dimension), dtype = DTYPE)

        # Assume discrete white noise model for the phase/phase velocity.
        self.state_cov[0:2, 0:2] = np.array([[.25, .5], [.5, 1.0]]) * phase_var
        self.state_cov[2:, 2:] = cov_basis_weights

        self.identity_cov = np.eye(self.state_cov.shape[0], dtype = DTYPE)

        self.basis_model = basis_model
        self.mean_basis_weights = mean_basis_weights.astype(DTYPE)
        self.cov_basis_weights = cov_basis_weights.astype(DTYPE)

    def get_measurement_model(self):
        """Computes the measurement model for an EKF.
        """
        measurement_model = np.zeros((self.measurement_dimension, self.state_dimension), dtype = DTYPE)

        f = lambda x, c: (np.exp(-(np.array([x - y for y in c], dtype = DTYPE) ** 2) / (2.0 * self.basis_model.scale)) * np.array([x - y for y in c], dtype = DTYPE)) * -(2.0 / (2.0 * self.basis_model.scale))

        basis_funcs = f(self.state_mean[0], self.basis_model.centers)

        for degree in range(self.measurement_dimension):
            offset = degree * self.basis_model.degree
            measurement_model[degree, 0] = np.dot(basis_funcs.T, self.state_mean[2 + offset: 2 + offset + self.basis_model.degree])

        basis_matrix = self.basis_model.get_block_diagonal_basis_matrix(np.asarray(self.state_mean[:1]), self.measurement_dimension)

        measurement_model[:, 2:] = basis_matrix.T

        return measurement_model

    def localize(self, measurement, measurement_noise):
        """Accepts a measurement matrix (vector of observed states) along with noise and iteratively updates the filter state.
        """
        for measurement_idx in range(measurement.shape[0]):
            # Assuming a constant velocity model, make our state prediction.
            self.state_mean[0] += self.state_mean[1]

            basis_matrix = self.basis_model.get_block_diagonal_basis_matrix(np.asarray(self.state_mean[:1]), self.measurement_dimension)

            predicted_measurement = np.dot(basis_matrix.T, self.state_mean[2:]).flatten()

            measurement_model = self.get_measurement_model()

            kalman_gain = np.dot(self.state_cov, measurement_model.T)
            kalman_gain = kalman_gain.dot(scipy.linalg.inv(np.dot(measurement_model, self.state_cov).dot(measurement_model.T) + measurement_noise))

            self.state_mean += np.dot(kalman_gain, measurement[measurement_idx] - predicted_measurement)

            self.state_cov = (self.identity_cov - np.dot(kalman_gain, measurement_model)).dot(self.state_cov)

        # Restrict final output to [0.0, 1.0] so it's a valid phase.
        if(self.state_mean[0] > 1.0):
            self.state_mean[0] = 1.0
        elif(self.state_mean[0] < 0.0):
            self.state_mean[0] = 0.0

        # Return phase and updated weights
        return self.state_mean[0], self.state_mean[2:]

class BayesianInteractionPrimitive(object):
    """The BayesianInteractionPrimitive class performs training and inference for a cooperative scenario.
    """
    def __init__(self, num_dof = 0, dof_names = [], basis_degree = 0):
        """The constructor function.
        """
        self.num_dof = num_dof
        self.dof_names = dof_names
        self.basis_degree = basis_degree

        self.basis_model = basis_model.GaussianBasisModel(self.basis_degree)
        self.basis_weights = []

        self.filter = None

    def export_data(self, file_name):
        """Exports the internal data to the given file name.
        """
        print("Exporting data to: " + str(file_name))

        data_struct = {
            "num_dof" : self.num_dof,
            "dof_names" : self.dof_names,
            "basis_degree" : self.basis_degree,
            "basis_weights" : self.basis_weights
        }

        with open(file_name, 'wb') as out_file:
            pickle.dump(data_struct, out_file, pickle.HIGHEST_PROTOCOL)

    def import_data(self, file_name):
        """Imports previously exported data.
        """
        print("Importing data from: " + str(file_name))

        with open(file_name, 'rb') as in_file:
            data_struct = pickle.load(in_file)

        self.num_dof = data_struct["num_dof"]
        self.dof_names = data_struct["dof_names"]
        self.basis_degree = data_struct["basis_degree"]
        self.basis_weights = data_struct["basis_weights"]

        self.basis_model = basis_model.GaussianBasisModel(self.basis_degree)

    def add_demonstration(self, trajectory):
        """Adds a demonstration to the BIP instance.
        """
        # N x M matrix, N is degrees of freedom, M is number of time steps
        if(type(trajectory) != np.ndarray):
            raise TypeError("Trajectory must be a numpy array.")

        if(len(trajectory) != self.num_dof):
            raise ValueError("Trajectory contains an invalid number of degrees of freedom.")

        self.basis_weights.append(self.get_basis_weights_linear(trajectory))

    def get_mean_trajectory(self, num_samples = DEFAULT_NUM_SAMPLES):
        """Gets the mean trajectory from the stored demonstrations.
        """
        mean, var = self.get_basis_weight_parameters()
        new_trajectory = np.zeros((self.num_dof, num_samples))

        domain = np.linspace(0, 1, num_samples, dtype = DTYPE)
        for idx in range(num_samples):
            basis_matrix = self.get_block_diagonal_basis_matrix(domain[idx : idx + 1])

            dist_mean = np.dot(basis_matrix.T, mean).flatten()

            new_trajectory[:, idx] = dist_mean

        return new_trajectory

    def get_approximate_trajectory(self, num_samples = DEFAULT_NUM_SAMPLES):
        """Gets the approximate trajectory for the last added demonstration.
        """
        trajectory = []
        domain = np.linspace(0, 1, num_samples, dtype = DTYPE)

        for idx in range(num_samples):
            basis_matrix = self.get_block_diagonal_basis_matrix(domain[idx : idx + 1])
            state = np.dot(basis_matrix.T, self.basis_weights[-1].flatten())

            for index in range(0, self.num_dof):
                if(len(trajectory) <= index):
                    trajectory.append([])

                trajectory[index].append(state[index])

        return np.array(trajectory)

    def get_approximate_trajectory_derivative(self, num_samples = DEFAULT_NUM_SAMPLES):
        """Gets the trajectory for the derivative of the last added demonstration.
        """
        trajectory = []

        domain = np.linspace(0, 1, num_samples, dtype = DTYPE)

        for idx in range(num_samples):
            basis_matrix = self.get_block_diagonal_basis_matrix_derivative(domain[idx : idx + 1])
            state = np.dot(basis_matrix.T, self.basis_weights[-1].flatten())

            for index in range(0, self.num_dof):
                if(len(trajectory) <= index):
                    trajectory.append([])

                trajectory[index].append(state[index])

        return trajectory

    def get_probability_distribution(self, num_samples = 100):
        """Gets the probability distribution for the currently added demonstrations.
        """
        mean, var = self.get_basis_weight_parameters()
        trajectory = np.zeros((self.num_dof, num_samples))
        upper_bound = np.zeros((self.num_dof, num_samples))
        lower_bound = np.zeros((self.num_dof, num_samples))

        domain = np.linspace(0, 1, num_samples, dtype = DTYPE)
        for idx in range(num_samples):
            basis_matrix = self.get_block_diagonal_basis_matrix(domain[idx : idx + 1])

            dist_mean = np.dot(basis_matrix.T, mean).flatten()
            dist_var = np.dot(np.dot(basis_matrix.T, var), basis_matrix)

            trajectory[:, idx] = dist_mean

            for dof_index in range(0, self.num_dof):
                std_dev = dist_var[dof_index][dof_index] ** 0.5

                upper_bound[dof_index, idx] = dist_mean[dof_index] + std_dev
                lower_bound[dof_index, idx] = dist_mean[dof_index] - std_dev

        return trajectory, upper_bound, lower_bound

    def get_basis_weights_linear(self, trajectory):
        """Computes weights for a linear basis model.
        """
        domain = np.linspace(0, 1, len(trajectory[0]), dtype = DTYPE)
        return self.basis_model.fit_basis_functions_linear_closed_form(domain, trajectory.T).T

    def get_basis_weight_parameters(self):
        """Computes metrics for the stored demonstrations.
        """
        weights = np.array(self.basis_weights, dtype = DTYPE).reshape((len(self.basis_weights), self.num_dof * self.basis_degree))
        mean = np.mean(weights, axis = 0)

        if(len(self.basis_weights) > 1):
            # np cov expects each row to represent a variable, and each column to represent an observation. So need to flip matrix.
            var = np.cov(weights.T)
        else:
            var = None

        return mean, var

    def get_block_diagonal_basis_matrix(self, time):
        """Gets the block diagonal basis matrix.
        """
        return self.basis_model.get_block_diagonal_basis_matrix(time, self.num_dof)

    def get_block_diagonal_basis_matrix_derivative(self, time):
        """Gets the derivative of the block diagonal basis matrix.
        """
        return self.basis_model.get_block_diagonal_basis_matrix_derivative(time, self.num_dof)

    def initialize_filter(self, phase_velocity = 0.01, phase_var = 0.000001):
        """Initializes the EKF SLAM filter.
        """
        mean, cov = self.get_basis_weight_parameters()
        self.filter = EKFSLAM(mean, cov, self.basis_model, self.num_dof, phase_velocity, phase_var)

    def generate_probable_trajectory_recursive(self, trajectory, observation_noise, num_samples = 100):
        """Updates the filter estimate given a partial observation.
        """
        if(self.filter is None):
            self.initialize_filter()

        phase, mean = self.filter.localize(trajectory.T, observation_noise)

        new_trajectory = np.zeros((self.num_dof, num_samples), dtype = DTYPE)

        # Create a sequence from the stored basis weights.
        domain = np.linspace(phase, 1, num_samples, dtype = DTYPE)

        for idx in range(num_samples):
            basis_matrix = self.get_block_diagonal_basis_matrix(domain[idx : idx + 1])

            dist_mean = np.dot(basis_matrix.T, mean).flatten()

            new_trajectory[:, idx] = dist_mean

        return new_trajectory, phase

    # Displays the probability that the current trajectory matches the stored trajectores at every instant in time.
    def plot_distribution(self, mean, upper_bound, lower_bound):
        """Plots a given probability distribution.
        """
        fig = plt.figure()
        for index in range(mean.shape[0]):
            new_plot = plt.subplot(mean.shape[0], 1, index + 1)
            domain = np.linspace(0, 1, mean.shape[1])

            new_plot.fill_between(domain, upper_bound[index], lower_bound[index], color = '#ccf5ff')
            new_plot.plot(domain, mean[index], color = '#000000')
            new_plot.set_title('Trajectory distribution for degree ' + self.dof_names[index])

        plt.show(block = False)

    def plot_trajectory(self, trajectory, observed_trajectory, mean_trajectory = None):
        """Plots a given trajectory.
        """
        fig = plt.figure()

        plt.plot(trajectory[0], trajectory[1])
        plt.plot(observed_trajectory[0], observed_trajectory[1])
        if(mean_trajectory is not None):
            plt.plot(mean_trajectory[0], mean_trajectory[1])

        fig.suptitle('Probable trajectory')

        fig = plt.figure()
        for index, degree in enumerate(trajectory):
            new_plot = plt.subplot(len(trajectory), 1, index + 1)

            domain = np.linspace(0, 1, len(trajectory[index]))
            new_plot.plot(domain, trajectory[index], label = "Generated")

            domain = np.linspace(0, 1, len(observed_trajectory[index]))
            new_plot.plot(domain, observed_trajectory[index], label = "Observed")

            if(mean_trajectory is not None):
                domain = np.linspace(0, 1, len(mean_trajectory[index]))
                new_plot.plot(domain, mean_trajectory[index], label = "Mean")

            new_plot.set_title('Trajectory for degree ' + self.dof_names[index])
            new_plot.legend()

        plt.show()

    def plot_partial_trajectory(self, trajectory, partial_observed_trajectory, mean_trajectory = None):
        """Plots a trajectory and a partially observed trajectory.
        """
        fig = plt.figure()

        start_partial = 0.0
        end_partial = float(partial_observed_trajectory.shape[1]) / (float(partial_observed_trajectory.shape[1]) + float(trajectory.shape[1]))

        plt.plot(trajectory[0], trajectory[1], "--", color = "#ff6a6a", label = "Generated", linewidth = 2.0)
        plt.plot(partial_observed_trajectory[0], partial_observed_trajectory[1], color = "#6ba3ff", label = "Observed", linewidth = 2.0)
        if(mean_trajectory is not None):
            plt.plot(mean_trajectory[0], mean_trajectory[1], color = "#85d87f", label = "Mean")

        fig.suptitle('Probable trajectory')
        plt.legend()

        plt.show()

    def plot_approximation(self, trajectory, approx_trajectory, approx_trajectory_deriv):
        """Plots a trajectory and its approximation.
        """
        domain = np.linspace(0, 1, len(trajectory[0]))
        approx_domain = np.linspace(0, 1, len(approx_trajectory[0]))

        for dof in range(self.num_dof):
            plt.figure()
            new_plot = plt.subplot(3, 1, 1)
            new_plot.plot(domain, trajectory[dof])
            new_plot.set_title('Original ' + self.dof_names[dof] + ' Data')

            new_plot = plt.subplot(3, 1, 2)
            # The trailing [0] is the dimension of the the state. In this case only plot position.
            new_plot.plot(approx_domain, approx_trajectory[dof])
            new_plot.set_title('Approximated ' + self.dof_names[dof] + ' Data')

            new_plot = plt.subplot(3, 1, 3)
            # The trailing [0] is the dimension of the the state. In this case only plot position.
            new_plot.plot(approx_domain, approx_trajectory_deriv[dof])
            new_plot.set_title('Approximated ' + self.dof_names[dof] + ' Derivative')

        plt.show()
