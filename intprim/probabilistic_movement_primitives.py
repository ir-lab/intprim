#!/usr/bin/python
##
#   This module defines the ProMP class, which is the user-facing class for deploying Probabilistic Movement Primitives.
#   The code for conditioning the ProMP is taken from the code by Sebastian Gomez-Gonzalez at https://github.com/sebasutp/promp
#	TODO: Implement the EM based learning with NIW prior
#
#   @author Vignesh Prasad <vignesh.prasad@tu-darmstadt.de>, TU Darmstadt
import constants
import scipy.linalg
import autograd.numpy as np
import pickle
import sklearn.preprocessing

##
#   The ProMP class is responsible for training an Probabilistic Movement Primitive model from demonstrations as well as performing run-time inference.
#   Support for importing and exporting a trained model as well
#
class ProMP(object):
	##
	#   The initialization method for ProMP.
	#
	#   @param basis_model The basis model corresponding to this state space.
	#   @param scaling_groups If provided, used to indicate which degrees of freedom should be scaled as a group.
	#
	def __init__(self, basis_model, scaling_groups = None):
		self.basis_model = basis_model
		self.scaling_groups = scaling_groups

		self.basis_weights = np.array([], dtype = intprim.constants.DTYPE)
		self.prior_fitted = False

		self.scalers = []
		# self.init_scalers()

	##
	#   Exports the internal state information from this model.
	#   Allows one to export a trained model and import it again without requiring training.
	#
	#   @param file_name The name of the export file.
	#
	def export_data(self, file_name):
		print("Exporting data to: " + str(file_name))

		data_struct = {
			"basis_weights" : self.basis_weights,
			"scaling_groups" : self.scaling_groups,
			"scalers" : self.scalers
		}

		with open(file_name, 'wb') as out_file:
			pickle.dump(data_struct, out_file, pickle.HIGHEST_PROTOCOL)

	##
	#   Imports the internal state information from an export file.
	#   Allows one to import a trained model without requiring training.
	#
	#   @param file_name The name of the import file.
	#
	def import_data(self, file_name):
		print("Importing data from: " + str(file_name))

		with open(file_name, 'rb') as in_file:
			data_struct = pickle.load(in_file)

		self.basis_weights = np.array(data_struct["basis_weights"])

		try:
			self.scaling_groups = data_struct["scaling_groups"]
			self.scalers = data_struct["scalers"]
		except KeyError:
			print("No scalers found during import!")

	##
	#   Internal method which initializes data scalers.
	#
	def init_scalers(self):
		if(self.scaling_groups is not None):
			for group in self.scaling_groups:
				self.scalers.append(sklearn.preprocessing.MinMaxScaler())

	##
	#   Iteratively fits data scalers.
	#   This must be called on all training demonstrations before fitting, if scaling is used.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#
	def compute_standardization(self, trajectory):
		if(len(self.scalers) > 0):
			# N x M matrix, N is degrees of freedom, M is number of time steps
			if(type(trajectory) != np.ndarray):
				raise TypeError("Trajectory must be a numpy array.")

			if(len(trajectory) != self.basis_model.num_observed_dof):
				raise ValueError("Trajectory contains an invalid number of degrees of freedom.")

			if(self.scaling_groups is not None):
				for group, scaler in zip(self.scaling_groups, self.scalers):
					scaler.partial_fit(trajectory[group, :].reshape(-1, 1))
		else:
			print("Skipping basis standardization...")

	##
	#   Iteratively adds a demonstration to the model.
	#   The demonstration is decomposed into the latent space and the weights are stored internally.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#
	def add_demonstration(self, trajectory):
		# N x M matrix, N is degrees of freedom, M is number of time steps
		if(type(trajectory) != np.ndarray):
			raise TypeError("Trajectory must be a numpy array.")

		if(len(trajectory) != self.basis_model.num_observed_dof):
			raise ValueError("Trajectory contains an invalid number of degrees of freedom. Got " + str(len(trajectory)) + " but expected " + str(self.basis_model.num_observed_dof))

		demonstration_weights = self.basis_transform(trajectory)

		if(self.basis_weights.shape[0] == 0):
			self.basis_weights = np.hstack([self.basis_weights, demonstration_weights])
		else:
			self.basis_weights = np.vstack([self.basis_weights, demonstration_weights])

	##
	#   Gets the mean trajectory of all trained demonstrations.
	#
	#   @param num_samples The length of the generated mean trajectory
	#
	#   @return mean_trajectory Matrix of dimension D x num_samples containing the mean trajectory.
	#
	def get_mean_trajectory(self, num_samples = intprim.constants.DEFAULT_NUM_SAMPLES):
		mean, var = self.get_basis_weight_parameters()

		domain = np.linspace(0, 1, num_samples, dtype = intprim.constants.DTYPE)

		return self.basis_inverse_transform(domain, mean)

	##
	#   Gets the approximated trajectory for the given demonstration.
	#   This is obtained by transforming the demonstration to the latent space and then projecting it back to measurement space.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#   @param num_samples The length of the generated approximate trajectory
	#
	#   @return approximate_trajectory Matrix of dimension D x num_samples containing the approximate trajectory.
	#
	def get_approximate_trajectory(self, trajectory, num_samples = intprim.constants.DEFAULT_NUM_SAMPLES, deriv = False):
		# N x M matrix, N is degrees of freedom, M is number of time steps
		if(type(trajectory) != np.ndarray):
			raise TypeError("Trajectory must be a numpy array.")

		if(len(trajectory) != self.basis_model.num_observed_dof):
			raise ValueError("Trajectory contains an invalid number of degrees of freedom.")

		basis_weights = self.basis_transform(trajectory)

		domain = np.linspace(0, 1, num_samples, dtype = intprim.constants.DTYPE)

		return self.basis_inverse_transform(domain, basis_weights, deriv)

	##
	#   Gets the approximated trajectory derivative for the given demonstration.
	#   This is obtained by transforming the demonstration to a latent space composed of the basis function derivatives and then projecting it back to measurement space.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#   @param num_samples The length of the generated approximate trajectory
	#
	#   @return approximate_trajectory Matrix of dimension D x num_samples containing the approximate trajectory.
	#
	def get_approximate_trajectory_derivative(self, trajectory, num_samples = intprim.constants.DEFAULT_NUM_SAMPLES):
		return self.get_approximate_trajectory(trajectory, num_samples, deriv = True)

	##
	#   Gets the probability distribution of the trained demonstrations.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#   @param num_samples The length of the generated distribution.
	#
	#   @return mean Matrix of dimension D x num_samples containing the mean of the distribution for every degree of freedom.
	#   @return upper_bound Matrix of dimension D x num_samples containing the mean + std of the distribution for every degree of freedom.
	#   @return lower_bound Matrix of dimension D x num_samples containing the mean - std of the distribution for every degree of freedom.
	#
	def get_probability_distribution(self, num_samples = intprim.constants.DEFAULT_NUM_SAMPLES):
		trajectory = np.zeros((self.basis_model.num_observed_dof, num_samples))
		upper_bound = np.zeros((self.basis_model.num_observed_dof, num_samples))
		lower_bound = np.zeros((self.basis_model.num_observed_dof, num_samples))

		domain = np.linspace(0, 1, num_samples, dtype = intprim.constants.DTYPE)
		for idx in range(num_samples):
			# In rare instances, projecting the covariance matrix can produce negative variance values in the diagonals of the projected matrix.
			# Therefore, we instead project each demonstration and manually calculate the empirical mean/covariance.
			projected_states = []
			for dem_idx in range(self.basis_weights.shape[0]):
				projected_states.append(self.basis_model.apply_coefficients(domain[idx], self.basis_weights[dem_idx, :]))
			projected_states = np.array(projected_states)

			dist_mean = np.mean(projected_states, axis = 0)
			dist_var = np.cov(projected_states.T)

			if(self.scaling_groups is not None):
				var_scale = np.ones(dist_mean.shape)
				for group, scaler in zip(self.scaling_groups, self.scalers):
					var_scale[group] = 1.0 / scaler.scale_
					dist_mean[group] = scaler.inverse_transform(dist_mean[group].reshape(-1, 1)).flatten()

				var_scale = np.diag(var_scale)
				dist_var = np.dot(var_scale, dist_var).dot(var_scale.T)

			trajectory[:, idx] = dist_mean

			for dof_index in range(0, self.basis_model.num_observed_dof):
				std_dev = dist_var[dof_index][dof_index] ** 0.5

				upper_bound[dof_index, idx] = dist_mean[dof_index] + std_dev
				lower_bound[dof_index, idx] = dist_mean[dof_index] - std_dev

		return trajectory, upper_bound, lower_bound

	##
	#   Transforms the given trajectory from measurement space into the latent basis space.
	#
	#   @param trajectory Matrix of dimension D x T containing a demonstration, where T is the number of timesteps and D is the dimension of the measurement space.
	#
	#   @return transformed_state Vector of dimension B containing the transformed trajectory.
	#
	def basis_transform(self, trajectory):
		if(self.scaling_groups is not None):
			scaled_trajectory = np.zeros(trajectory.shape)

			for group, scaler in zip(self.scaling_groups, self.scalers):
				scaled_trajectory[group, :] = scaler.transform(trajectory[group, :].reshape(-1, 1)).reshape(trajectory[group, :].shape)

			trajectory = scaled_trajectory

		domain = np.linspace(0, 1, len(trajectory[0]), dtype = intprim.constants.DTYPE)
		return self.basis_model.fit_basis_functions_linear_closed_form(domain, trajectory.T)

	##
	#   Transforms the given basis space weights to measurement space for the given phase values.
	#
	#   @param x Vector of dimension T containing the phase values that the basis space weights should be projected at.
	#   @param weights Vector of dimension B containing the basis space weights.
	#   @param deriv True if the basis weights should be transformed with basis function derivatives, False for normal basis functions.
	#
	#   @return transformed_trajectory Matrix of dimension D x T containing the transformed trajectory.
	#
	def basis_inverse_transform(self, x, weights, deriv = False):
		trajectory = np.zeros((self.basis_model.num_observed_dof, x.shape[0]), dtype = intprim.constants.DTYPE)

		for idx in range(x.shape[0]):
			trajectory[:, idx] = self.basis_model.apply_coefficients(x[idx], weights, deriv)

		if(self.scaling_groups is not None):
			for group, scaler in zip(self.scaling_groups, self.scalers):
				trajectory[group, :] = scaler.inverse_transform(trajectory[group, :].reshape(-1, 1)).reshape(trajectory[group, :].shape)

		return trajectory

	##
	#   Gets the mean and covariance for the trained demonstrations.
	#
	#   @return mean Vector of dimension B containing the sample mean of the trained basis weights.
	#   @return var Matrix of dimension B x B containing the sample covariance of the trained basis weights.
	#
	def get_basis_weight_parameters(self):
		mean = np.mean(self.basis_weights, axis = 0)

		if(self.basis_weights.shape[0] > 1):
			var = np.cov(self.basis_weights, rowvar = False)
		else:
			var = None

		return mean, var

	##
	#   Performs inference over the given time duration returns a probable trajectory.
	#
	#   @param times Vector of dimension num_samples containing times at which to generate the trajectory.
	#   @paran mean Vector of dimension B containing the sample mean of the basis weights.
	#   @param var Matrix of dimension B x B containing the sample covariance of the basis weights.
	#
	#   @return new_trajectory Matrix of dimension D x num_samples containing the inferred trajectory.
	#   @return weights Vector of dimension B containing the weights used to infer the trajectory.
	#
	def generate_probable_trajectory(self, times, mean=None, var=None):
		_mean, _var = self.get_basis_weight_parameters()
		if mean==None:
			mean = _mean
		if var==None:
			var = _var

		weights = np.random.multivariate_normal(mean, var)
		new_trajectory = self.basis_inverse_transform(times, weights)

		return new_trajectory, weights

	##
	#   Get the PromP weights after conditioning to reach a particular joint configuration.
	#
	#   @param t Time Phase at which the required joint configuration should be reached.
	#   @paran mean_q Vector of dimension D containing the mean of the required joint configuration to be reached.
	#   @param var_q Matrix of dimension B x B containing the sample covariance of the required joint configuration to be reached.
	#   @paran mean_w Vector of dimension B containing the sample mean of the basis weights.
	#   @param var_w Matrix of dimension B x B containing the sample covariance of the basis weights.
	#
	#   @return mean_w Vector of dimension B containing the sample mean of the basis weights after conditioning.
	#   @return var_w Matrix of dimension B x B containing the sample covariance of the basis weights after conditioning.
	#
	def get_conditioned_weights(self, t, mean_q, var_q=None, mean_w=None, var_w=None):
		basis_funcs = self.basis_model.get_block_diagonal_basis_matrix(t)
		d,lw = basis_funcs.shape
		_mean_w, _var_w = self.get_basis_weight_parameters()
		if mean_w==None:
			mean_w = _mean_w
		if var_w==None:
			var_w = _var_w

		tmp1 = np.dot(var_w, basis_funcs)
		tmp2 = np.dot(basis_funcs.T, np.dot(var_w, basis_funcs))
		tmp2 = np.linalg.inv(tmp2)
		tmp3 = np.dot(tmp1,tmp2)
		mean_w = mean_w + np.dot(tmp3, (mean_q - np.dot(basis_funcs.T, mean_w)))
		tmp4 = np.eye(lw)
		if var_q is not None:
			tmp4 -= np.dot(var_q, tmp2)
		var_w = var_w - np.dot(tmp3, np.dot(tmp4, tmp1.T))

		return mean_w, var_w

	##	#   Get the marginal distribution of the learnt trajectory at a given time.
	#
	#   @param t Time Phase at which the marginal distribution is to be calculated.
	#   @paran mean_w Vector of dimension B containing the sample mean of the basis weights.
	#   @param var_w Matrix of dimension B x B containing the sample covariance of the basis weights.
	#
	#   @return mean_q Vector of dimension D containing the mean of the marginal distribution at the given time.
	#   @return var_q Matrix of dimension B x B containing the sample covariance of the marginal distribution at the given time.
	#
	def get_marginal(self, t, mean_w=None, var_w=None):
		basis_funcs = self.basis_model.get_block_diagonal_basis_matrix(t/T)
		d,lw = basis_funcs.shape
		_mean_w, _var_w = self.get_basis_weight_parameters()
		if mean_w==None:
			mean_w = _mean_w
		if var_w==None:
			var_w = _var_w

		var_q = np.dot(basis_funcs.T, np.dot(var_w, basis_funcs))
		mean_q = np.dot(basis_funcs.T, mean_w)

		return mean_q, var_q
