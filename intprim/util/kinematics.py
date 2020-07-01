##
#   This module defines the base class for implementing the forward kinematics of your robot.
#   The code taken from the code by Sebastian Gomez-Gonzalez at https://github.com/sebasutp/promp
#
#   @author Vignesh Prasad <vignesh.prasad@tu-darmstadt.de>, TU Darmstadt
import numpy as np
import scipy.optimize as opt

##
#   Base Forward kinematics object implementing basic forward and inverse kinematics functionality taken from https://github.com/sebasutp/promp. For your own robot, please extend this class and define your own `_link_matrices` function
#
class BaseKinematicsClass:

	##
	#	Initialization function for the class. When extending, add whatever parameters are necessary for implementing the `_link_matrices` function, such as end effector pose or arm lengths etc.
	#
	def __init__(self):
		pass

	##
	#	Function implementing the relative linkwise forward kinematics of the robot. This is the main function that needs to be implemented while extending this base class.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return T list of N + 2 Transformation Matrices each  having dimensions 4 x 4. The first should be a transformation to the base link of the robot. The next N matrices are the result of applying each of the N degrees of freedom. The ith matrix is the relative transformation between the (i-1)th to the ith frame of reference  The last matrix if the transformation to the final end effector link from the last frame of reference.
	#
	def _link_matrices(self,q):
		base_link_transform = np.eye(4)
		link_transforms = [np.eye(4)]*len(q)
		end_effector_transform = np.eye(4)

		return [base_link_transform] + link_transforms + [end_effector_transform]

	##
	#	Function implementing the forward kinematics of the robot in the global frame of reference.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return absolute_transforms list of N + 1 Transformation Matrices each  having dimensions 4 x 4. The first N matrices represent the pose of the reference frames of each of the N degrees of freedon in the global frame. The ith matrix is the relative transformation between the global frame to the ith frame of reference  The last matrix if the transformation to the final end effector link.
	#
	def forward_kinematics(self,q):
		H = self._link_matrices(q)
		A = H[0]
		absolute_transforms = []
		for i in range(1,len(H)):
			A = np.dot(A,H[i])
			absolute_transforms.append(A)
		return absolute_transforms

	##
	#	Function to obtain the euler angles from a given rotation matrix.
	#
	#   @param rotMat The input rotation matrix.
	#
	#   @return eul Vector of size 3 with the yaw, pitch and roll angles respectively.
	#
	def __rotMatToEul(self,rotMat):
		eul = np.zeros(3)
		eul[0] = np.arctan2(-rotMat[2,1],rotMat[2,2])
		eul[1] = np.arctan2(rotMat[2,0],np.sqrt(rotMat[2,1]**2+rotMat[2,2]**2))
		eul[2] = np.arctan2(-rotMat[1,0],rotMat[0,0])
		return eul

	##
	#	Function to obtain 6DoF pose of the end effector.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Vector of size 3 with the x,y,z positions of the end effector.
	#   @return orientation Vector of size 3 with the yaw, pitch and roll angles of the end effector.
	#
	def end_effector(self, q ,As=None):
		if As is None:
			As = self.forward_kinematics(q)
		end_eff = As[-1]
		pos = end_eff[0:3,3]
		orientation = self.__rotMatToEul(end_eff[0:3,0:3].transpose())
		return pos, orientation

	##
	#	Calculates the numerical jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param eps Amount to perturb each degree of freedom with to calculate the jacobian.
	#
	#   @return jac Jacobian matrix of the x,y,z positions of the end effector having shape 3 x N.
	#
	def __num_jac(self, q, eps = 1e-6):
		jac = np.zeros((3,len(q)))
		fx,ori = self.end_effector(q)
		for i in range(len(q)):
			q[i] += eps
			fxh,ori = self.end_effector(q)
			jac[:,i] = (fxh - fx) / eps
			q[i] -= eps
		return jac

	##
	#	Calculates the analytical jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param As list of N + 1 Transformation Matrices each  having dimensions 4 x 4. The first N matrices  represent the pose of the reference frames of each of the N degrees of freedon in the global frame. The ith matrix is the relative transformation between the global frame to the ith frame of reference  The last matrix if the transformation to the final end effector link.
	#
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#
	def __analytic_jac(self, q, As=None):
		jac = np.zeros((6,len(q)))
		if As is None:
			As = self.forward_kinematics(q)
		pe = As[-1][0:3,3]
		for i in range(len(q)):
			zprev = As[i][0:3,2]
			pprev = As[i][0:3,3]
			jac[0:3,i] = np.cross(zprev, pe - pprev)
			jac[3:6,i] = zprev
		return jac

	##
	#	Calculates the jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#
	def jacobian(self, q):
		As = self.forward_kinematics(q)
		#jac = self.__num_jac(q,1e-6)
		jac = self.__analytic_jac(q, As)
		return jac

	##
	#	Calculates the position of the end effector and the jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Vector of size 3 with the x,y,z positions of the end effector.
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#   @return ori Vector of size 3 with the yaw, pitch and roll angles of the end effector.
	#
	def position_and_jac(self, q):
		As = self.forward_kinematics(q)
		jac = self.__analytic_jac(q, As)
		pos, ori = self.end_effector(q, As)
		return pos, jac, ori

	##
	#	Calculates the trajectory of the end effector given a set of joint configuration trajectories.
	#
	#   @param Q Matrix of dimension num_samples x N containing the trajectories of each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Matrix of size num_samples x 3 containing the the x,y,z locations of the end effector.
	#   @return orientation Matrix of size num_samples x 3 with the yaw, pitch and roll angles of the end effector at each pointin the trajectory.
	#
	def end_eff_trajectory(self, Q):
		pos = []
		orientation = []
		for t in range(len(Q)):
			post, ort = self.end_effector(Q[t])
			pos.append(post)
			orientation.append(ort)
		return np.array(pos), np.array(orientation)

	##
	#	Calculates the cost function and gradient of the cost function for the Inverse Kinematics optimization.
	#
	#   @param theta Vector of dimension N containing the current estimate of values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param mu_theta Vector of dimension N containing the mean values of the prior for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param inv_sigma_theta Matrix of dimension N x N containing the inverse of the covariance matrix of the prior for each degree of freedom of the robot.
	#   @param mu_x Vector of dimension 3 containing the mean values of the distribution for the goal's 3D coordinates in meters.
	#   @param inv_sigma_x Matrix of dimension 3 x 3 containing the inverse of the covariance matrix of the distribution for the goal's 3D coordinates.
	#
	#   @return nll Scalar value returning the current cost function value.
	#   @return grad_nll Vector of dimension N containing the gradient of the cost function w.r.t. the current estimate theta.
	#
	def __laplace_cost_and_grad(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x):
		f_th, jac_th, ori = self.position_and_jac(theta)
		jac_th = jac_th[0:3,:]
		diff1 = theta - mu_theta
		tmp1 = np.dot(inv_sigma_theta, diff1)
		diff2 = f_th - mu_x
		tmp2 = np.dot(inv_sigma_x, diff2)
		nll = 0.5*(np.dot(diff1,tmp1) + np.dot(diff2,tmp2))
		grad_nll = tmp1 + np.dot(jac_th.T,tmp2)

		return nll, grad_nll

	##
	#	Solves the Inverse Kinematics of the robot to reach a particular 3D location while trying to stick close to a prior distribution of the joint configuration.
	#
	#   @param mu_theta Vector of dimension N containing the mean values of the prior for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param sig_theta Matrix of dimension N x N containing the covariance matrix of the prior for each degree of freedom of the robot.
	#   @param mu_x Vector of dimension 3 containing the mean values of the distribution for the goal's 3D coordinates in meters.
	#   @param sig_x Matrix of dimension 3 x 3 containing the covariance matrix of the distribution for the goal's 3D coordinates.
	#
	#   @param pos_mean Vector of dimension N containing the mean values of the posterior for each degree of freedom of the robot in radians (for angles) or meters (for translations) after solving the inverse kinematics.
	#   @param pos_cov Matrix of dimension N x N containing the covariance matrix of the posterior for each degree of freedom of the robot after solving the inverse kinematics.
	#
	def inv_kin(self, mu_theta, sig_theta, mu_x, sig_x, **kwargs):
		inv_sig_theta = np.linalg.inv(sig_theta)
		inv_sig_x = np.linalg.inv(sig_x)
		cost_grad = lambda theta: self.__laplace_cost_and_grad(theta, mu_theta, inv_sig_theta, mu_x, inv_sig_x)
		cost = lambda theta: cost_grad(theta)[0]
		grad = lambda theta: cost_grad(theta)[1]

		kwargs.setdefault('method', 'BFGS')
		kwargs.setdefault('jac', grad)
		res = opt.minimize(cost, mu_theta, **kwargs)
		post_mean = res.x
		if hasattr(res, 'hess_inv'):
			post_cov = res.hess_inv
		else:
			post_cov = None
		print(res)
		return post_mean, post_cov
