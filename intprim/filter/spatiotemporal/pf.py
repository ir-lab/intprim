##
#   This module defines a spatiotemporal filter based off of the particle filter.
#
#   @author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
import matplotlib.animation
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import scipy.linalg
import scipy.stats

import intprim.constants
from intprim.filter.spatiotemporal import nonlinear_system

##
#   The ParticleFilter class localizes an interaction in time and space via sequential Monte Carlo sampling.
#   This class is a recursive filter, meaning it maintains state information between successive calls to localize().
#   As with the other spatiotemporal filters, the EnsembleKalmanFilter's internal state consists of (N+1) dimensions modeling
#   the N-th order phase system plus B dimensions modeling the latent space of the interaction.
#   However, unlike the ExtendedKalmanFilter, this class does not maintain a single explicit state instance.
#   Instead, an internal ensemble of E members is maintained, such that the ensemble matrix dimension is (N+1)+B x E.
#   As a result, no explicit covariance matrix is maintained. Instead, the covariance is modeled implicitly in the
#   ensemble via the sample covariance.
#
#   Note: this class is *not* optimized. In particular, the calculation of the PDF in localize is done individually for each ensemble member which is costly.
#
#   References:\n
#   Campbell, J., Stepputtis, S., & Ben Amor, H. (2019). Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks.\n
#   https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
#
class ParticleFilter(nonlinear_system.NonLinearSystem):
    ##
    #   The initialization method for the ParticleFilter. Responsible for creating the initial ensemble.
    #
    #   @param basis_model The basis model corresponding to this state space.
    #   @param phase_mean Vector of dimension N corresponding to the initial mean of the constant velocity phase system. If this is a 0th order system, should contain only [phase_mean]. If this is 1st order, should contain [phase_mean, phase_velocity]. If this is 2nd order, should contain [phase_mean, phase_velocity, phase_acceleration]. Anything above 2nd order is not supported.
    #   @param phase_var Vector of dimension N corresponding to the diagonal of the initial covariance of the constant velocity phase system. If this is a 0th order system, should contain only [phase_var]. If this is 1st order, should contain [phase_var, phase_velocity_var]. If this is 2nd order, should contain [phase_var, phase_velocity_var, phase_acceleration_var]. Anything above 2nd order is not supported.
    #   @param proc_var The process noise of the constant velocity phase system. This is a scalar value corresponding to the variance of a piecewise white noise model.
    #   @param initial_ensemble Matrix of dimension E x B which represents an initial ensemble. Typically, this is just the set of basis weights corresponding to E training demonstrations.
    #   @param num_effective_ratio If the number of effective samples falls below this ratio, then resampling is performed. Default is 0.5 which means resampling occurs when less than 50% of the samples are ineffective.
    #   @param time_delta The amount of time that elapses between time steps. This serves as a scaling factor to the constant velocity phase system. In most cases, this should be set to 1.0.
    #   @param cyclical Indicates whether this is a cyclical primitive. If True, the internal phase state will cycle back to 0 once it exceeds 1, allowing for continuous inference of periodic interactions.
    #
    def __init__(self,
                 basis_model,
                 initial_phase_mean,
                 initial_phase_var,
                 proc_var,
                 initial_ensemble,
                 num_effective_ratio = 0.5,
                 time_delta = 1.0,
                 cyclical = False):
        super(ParticleFilter, self).__init__(basis_model, proc_var, time_delta, len(initial_phase_mean) - 1)

        self.ensemble_size = initial_ensemble.shape[0]
        self.cyclical = cyclical

        initial_phase_mean = np.array(initial_phase_mean, dtype = intprim.constants.DTYPE)
        initial_phase_var = np.diag(initial_phase_var).astype(intprim.constants.DTYPE)

        self.ensemble = np.zeros((self.state_dimension, self.ensemble_size))
        self.ensemble[:self.system_size, :] = np.random.multivariate_normal(initial_phase_mean[:self.system_size], initial_phase_var[:self.system_size, :self.system_size], size = self.ensemble_size).T

        self.ensemble[self.system_size:, :] = initial_ensemble.T

        # print("Ensemble shape: " + str(self.ensemble.shape))

        self.weights = np.ones(self.ensemble_size) / self.ensemble_size

        self.num_effective_threshold = int(num_effective_ratio * self.ensemble_size)

    ##
    #   Calculates the unbiased sample mean of the ensemble.
    #   The formula used to compute the mean is: $$ \\bar{\\boldsymbol{X}} = \\frac{1}{E} \\sum_{j=1}^{E} \\boldsymbol{x}^{j}, $$
    #   where \f$ \boldsymbol{x}^{j} \in \mathbb{R}^{N+1+B} \f$ and \f$ \bar{\boldsymbol{X}} \in \mathbb{R}^{N+1+B}. \f$
    #
    #   @param ensemble The ensemble from which to calculate the sample mean. If None, uses the internal state ensemble.
    #
    #   @returns Vector of dimension N+1+B containing the sample mean.
    #
    def get_ensemble_mean(self, ensemble = None):
        if(ensemble is None):
            return np.sum(self.ensemble, axis = 1) / self.ensemble_size
        else:
            return np.sum(ensemble, axis = 1) / self.ensemble_size

    ##
    #   Calculates the unbiased sample covariance of the ensemble.
    #   The formula used to compute the covariance is: $$ cov(\\boldsymbol{X}) = \\frac{1}{E - 1} \\boldsymbol{A} \\boldsymbol{A}^{T}, \\qquad \\boldsymbol{A} = \\boldsymbol{X} - \\bar{\\boldsymbol{X}}, $$
    #   where \f$ cov(\boldsymbol{X}) \in \mathbb{R}^{N+1+B \times N+1+B}. \f$
    #
    #   @param ensemble The ensemble from which to calculate the sample covariance. If none, uses the internal state ensemble.
    #
    #   @returns Matrix of dimension N+1+B x N+1+B containing the sample covariance.
    #
    def get_ensemble_covariance(self, ensemble = None):
        mean = self.get_ensemble_mean(ensemble)

        if(ensemble is None):
            deviation = (self.ensemble.T - mean).T
        else:
            deviation = (ensemble.T - mean).T

        # Ensemble variance is the square deviation divided by the ensemble size - 1. [1]
        return np.dot(deviation, deviation.T) / (self.ensemble_size - 1.0)

    ##
    #   Calculates the unbiased sample mean and covariance of the ensemble projected into the observation space.
    #   This is computed by first projecting the entire ensemble to measurement space then calculating the sample covariance.
    #   Mathematically, calculating the sample covariance and then projecting it should be equivalent but numerical issues sometimes arise and so this method is preferred.
    #   The formula used to compute the covariance is: $$ cov(\\boldsymbol{H} \\boldsymbol{X}) = \\frac{1}{E - 1} (\\boldsymbol{H}\\boldsymbol{A}) (\\boldsymbol{H} \\boldsymbol{A}^{T}), \\qquad \\boldsymbol{H} \\boldsymbol{A} = \\boldsymbol{X} - \\frac{1}{E} \\sum_{j=1}^{E}h(\\boldsymbol{x}^j), $$
    #   where \f$ cov(\boldsymbol{H} \boldsymbol{X}) \in \mathbb{R}^{D \times D} \f$ and \f$ h(\cdot) \f$ is the nonlinear observation function.
    #
    #   @param phase The phase value \f$ \phi \f$ to use for the projection.
    #   @param ensemble The ensemble from which to calculate the projected mean and covariance. If none, uses the internal state ensemble.
    #
    #   @returns Vector of dimension D containing the sample mean, matrix of dimension D x D containing the sample covariance.
    #
    def get_projected_mean_covariance(self, phase, ensemble = None):
        if(ensemble is None):
            ensemble = self.ensemble

        if(phase is None):
            orig_mean = self.get_ensemble_mean(ensemble)
            phase = orig_mean[0]

        # Project ensemble to measurement dimension
        hx_matrix = np.zeros((self.measurement_dimension, self.ensemble_size), dtype = intprim.constants.DTYPE)
        for index in range(self.ensemble_size):
            hx_matrix[:, index] = self.basis_model.apply_coefficients(phase, ensemble[self.system_size:, index])

        # Calculate ensemble covariance.
        # Could calculate in state space then project, but should be the same.
        mean = np.sum(hx_matrix, axis = 1) / self.ensemble_size
        deviation = (hx_matrix.T - mean).T
        # Ensemble variance is the square deviation divided by the ensemble size - 1. [1]
        covariance = np.dot(deviation, deviation.T) / (self.ensemble_size - 1.0)

        return mean, covariance, hx_matrix

    ##
    #   The nonlinear observation function \f$ h(\cdot) \f$ which maps an input state of dimension \f$ \mathbb{R}^{B} \f$ to an output observation of dimension \f$ \mathbb{R}^D. \f$
    #   This is simply the dot product of the basis functions to the corresponding basis weights.
    #   The nonlinearity comes from the usage of the phase state variable as a parameter to the basis functions.
    #
    #   @param state Vector of dimension N+1+B which is the state to be projected.
    #
    #   @returns Vector of dimension D containing the projected state.
    #
    def h(self, state):
        return self.basis_model.apply_coefficients(state[0], state[self.system_size:])

    ##
    #   This method computes the projection of the ensemble to the observation space.
    #   Each member is projected such that the input ensemble is mapped from \f$ \mathbb{R}^{N+1+B \times E} \f$ to \f$ \mathbb{R}^{D \times E} \f$.
    #   The formula used to compute this is: $$ \\boldsymbol{H} \\boldsymbol{X} = \\left[h(\\boldsymbol{x}^1), \\dots, h(\\boldsymbol{x}^E) \\right]. $$
    #
    #   @param output_matrix Matrix of dimension D x E in which the results should be stored.
    #
    def hx(self, output_matrix):
        for index in range(self.ensemble_size):
            output_matrix[:, index] = self.h(self.ensemble[:, index])

    ##
    #   A systematic resampling scheme for the ensemble.
    #   Divide the ensemble weights into N regions, then choose a sample from each region such that all samples are exactly 1/N distance apart.
    #   Reference: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
    #
    #   @returns Vector of dimension E containing indices of the ensemble members to resample.
    #
    def systematic_resample(self):
        # Algorithm credit goes to: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

        positions = (np.arange(self.ensemble_size) + np.random.random()) / self.ensemble_size

        indices = np.zeros(self.ensemble_size, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i = 0
        j = 0
        while i < self.ensemble_size:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    ##
    #   This method performs simultaneous localization of both time (phase) and space (basis weights).
    #   This is a recursive call, which means it updates the internal state estimate based on the given observations.
    #   For each observation given, two steps are performed recursively:
    #   First, the current ensemble is propagated forward in time in what is known as the prediction step.
    #   $$ \\boldsymbol{x}^j_{t|t-1} = \\boldsymbol{G} \\boldsymbol{x}^j_{t-1|t-1} + \\mathcal{N} \\left(0, \\boldsymbol{Q}_t\\right), \\quad 1 \\leq j \\leq E. $$
    #   Note that in this case, we only apply the dot product to the first N+1 dimensions of the state. This is for computational efficiency as only the constant velocity phase system has a non-zero transition.
    #
    #   Next, we integrate the observations into the current ensemble in the update step.
    #   This involves calculating a weight for each ensemble member based on the likelihood of the measurement having been produced.
    #   The ensemble is resampled according to a systematic scheme if the number of effective members is less than a threshold.
    #
    #   Lastly, the sample mean and covariance of the ensemble are returned.
    #   At the end of both the prediction and update steps the internal phase value is clipped such that it falls within the range [0, 1].
    #
    #   @param measurement Matrix of dimension T x D containing observations, where T is the number of timesteps that have been observed since the last call to localize() and D is the dimension of the measurement space.
    #   @param measurement_noise Matrix of dimension D x D containing the measurement noise for the given set of measurements.
    #   @param active_dofs Vector of dimension \f$ D_o \f$ containing measurement space indices of the observed degrees of freedom. Note that the measurements will also contain unobserved degrees of freedom, but their values should not be used for inference.
    #   @param return_phase_variance True if the mean/variance for the phase system should be returned in addition to the basis weights.
    #
    #   @returns Scalar value containing the inferred phase, Vector of dimension D (or N+1+D if return_phase_variance is True) containing inferred mean, Matrix of dimension D x D (or N+1+D x N+1+D if return_phase_variance is True).
    def localize(self, measurement, measurement_noise, active_dofs, return_phase_variance = False):
        transition_model = self.get_transition_model()
        hx_matrix = np.zeros((self.measurement_dimension, self.ensemble_size), dtype = intprim.constants.DTYPE)

        for measurement_idx in range(measurement.shape[0]):
            # Make forward prediction for each ensemble member
            for index in range(self.ensemble_size):
                self.ensemble[:self.system_size, index] = np.dot(transition_model[:self.system_size, :self.system_size], self.ensemble[:self.system_size, index])

            # Project the phase values to be bounded by [0, 1].
            self.ensemble[0, self.ensemble[0, :] > 1.0] = 1.0
            self.ensemble[0, self.ensemble[0, :] < 0.0] = 0.0

            # Add process noise to predictions
            self.ensemble[:self.system_size, :] += np.random.multivariate_normal(np.zeros(self.system_size), self.phase_process_noise, size = self.ensemble_size, check_valid = "ignore").T

            # Calculate HX
            self.hx(hx_matrix)

            # Calculate weights for particles.
            for particle_idx in range(self.ensemble_size):
                self.weights[particle_idx] *= scipy.stats.multivariate_normal.pdf(measurement[measurement_idx], hx_matrix[:, particle_idx], measurement_noise)

            # Add insignificant amount to avoid numerical issues.
            self.weights += 1.e-300
            self.weights /= sum(self.weights)

            num_effective = 1.0 / np.sum(np.square(self.weights))

            if(num_effective < self.num_effective_threshold):
                # Resample particles with weights.
                indices = self.systematic_resample()
                self.ensemble[:, :] = self.ensemble[:, indices]
                self.weights[:] = self.weights[indices]
                self.weights.fill(1.0 / self.ensemble_size)

            # If the interaction is cyclical and the mean is greater than 1.0, then subtract 1.0 from all ensemble members.
            # Some may be less than 0, but that is ok since the mean is still greater than 0.
            if(self.cyclical):
                mean = self.get_ensemble_mean()
                if(mean[0] >= 1.0):
                    self.ensemble[0, :] -= 1.0

        # Project the phase values to be bounded by [0, 1].
        self.ensemble[0, self.ensemble[0, :] > 1.0] = 1.0
        self.ensemble[0, self.ensemble[0, :] < 0.0] = 0.0

        expected_value = self.get_ensemble_mean()
        expected_variance = self.get_ensemble_covariance()


        if(return_phase_variance is False):
            return expected_value[0], expected_value[self.system_size:], expected_variance[self.system_size:, self.system_size:]
        else:
            return expected_value[0], expected_variance[0, 0], expected_value[self.system_size:], expected_variance[self.system_size:, self.system_size:]
