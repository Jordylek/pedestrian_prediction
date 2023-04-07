from __future__ import division

import numpy as np
from .classic import GridWorldMDP, transition_helper, MDP2D, Directions, ActionConverter


def dist(a, b):
	return np.linalg.norm(a-b)


def compute_confidence_set(occupancy, p):
	"""
	Compute the confidence set of size 1-p for a given occupancy probability vector.
	:param occupancy: occupancy probability vector of size S or matrix of size (H,T, S) where H is the number of humans and T the time horizon
	:param p: uncertainty level.
	:return: confidence set of size 1-p
	"""
	p = np.clip(p, 0, 1)
	if np.ndim(occupancy) == 1:
		occupancy = occupancy[np.newaxis, :]
	H = occupancy.shape[0]
	sorted_states = np.argsort(occupancy)[:, ::-1]
	sorted_probs = occupancy[np.arange(H)[:, None], sorted_states]
	cumulative_probs = np.cumsum(sorted_probs, axis=1)
	max_index = np.argmax(cumulative_probs >= 1 - p, axis=1)
	confidence_set = [sorted_states[h, :max_index[h]+1] for h in range(H)]
	# sorted_states[np.arange(H), np.argmax(cumulative_probs >= 1 - p, axis=1)]
	return confidence_set


def compute_largest_uncertainty_level(occupancy, state):
	"""
	Compute the smallest size of the confidence set containing a given state.
	:param occupancy: occupancy probability vector of size S
	:param state: state
	:return: smallest size of the confidence set containing the state
	"""
	if np.ndim(occupancy) == 1:
		occupancy = occupancy[np.newaxis, :]
		state = np.array([state])
	H = occupancy.shape[0]
	lam_per_human = np.zeros(H)
	for h in range(H):
		sorted_states = np.argsort(occupancy)[h, ::-1]
		s_idx = np.where(sorted_states == state[h])[0][0]
		lam_per_human[h] = 1 - np.sum(occupancy[h, sorted_states[:s_idx + 1]])
	return np.max(lam_per_human)


class GridWorldExpandedAdaptive(GridWorldMDP):

	def __init__(self, rows, cols, list_occupancy_probability_human, human_traj, T, kappa=50, gamma=0.9,
				 max_step_size=3, **kwargs):
		"""
		:param rows: number of rows of GridWorld
		:param cols: number of columns of GridWorld
		:param list_occupancy_probability_human: list of occupancy probabilities for each time step. It is a list of size T,
		and element t is a matrix of size (H, T-t, S) where H is the number of humans
		:param human_traj: matrix of size (H, T). For each time step, it contains the predicted position of the human
		:param T: number of time steps
		:param kappa: parameter for the reward function
		:param gamma: discount factor
		:param max_step_size: maximum number of steps allowed for the agent.
		:param kwargs:
		"""

		self.T = T  # number of time steps
		self.kappa = kappa
		self.gamma = gamma
		self.max_step_size = max_step_size
		# self.penalty_type = penalty_type
		# assert self.penalty_type in ['hard', 'soft']
		# self.reward_dict = self._build_reward_dict(closeness=0.5)
		super().__init__(rows=rows, cols=cols, max_step_size=max_step_size, **kwargs)
		self.list_occupancy_probability_human = list_occupancy_probability_human
		self.H = human_traj.shape[0]  # number of humans
		self.human_traj = human_traj
		self.distances = self.distances_between_states()
		self.reward_cache = {}
		self.transitions = self.transition_cached  # deterministic transitions: T[s,a] = s'

	def distances_between_states(self):
		"""
		Compute the distance between each pair of states
		:return: matrix of size (S, S) with distances
		"""
		distances = np.zeros((self.S, self.S))
		for s in range(self.S):
			for s_prime in range(s, self.S):
				distances[s, s_prime] = dist(self.state_to_real_coor(s), self.state_to_real_coor(s_prime))
				distances[s_prime, s] = distances[s, s_prime]
		return distances

	def _dist_reward(self, s, a, s_prime, goal_state, goal_stuck):
		"""
		Compute the reward for the given state, action, next state for getting closer to the goal
		:param s: current state
		:param a: action
		:param s_prime: next state
		:param goal_state: goal state
		:param goal_stuck: goal stuck
		:return: reward
		"""
		if goal_stuck:
			if s == goal_state:
				if s_prime == goal_state:
					return 0
				else:
					return -np.inf
		reward = self.distances[s, goal_state] - self.distances[s_prime, goal_state]
		return reward

	def reset_occupancy_probability_human(self):
		self.occupancy_probability_human = self.list_occupancy_probability_human[0]

	def update_occupancy_probability_human(self, t):
		self.occupancy_probability_human = self.list_occupancy_probability_human[t]

	def closeness_penalty(self, s_prime, t, lam=0.05, penalty_type='hard'):
		"""
		Compute the penalty for being at s_prime t steps from now given the current information about the human.
		:param s_prime: next state
		:param t: timestep in the future
		:param lam: parameter for the penalty
		:param penalty_type: type of penalty
		:return: penalty
		"""
		human_probabilities = self.occupancy_probability_human[:, t]
		assert penalty_type in ['hard', 'soft', 'ignore']
		if penalty_type == 'hard':
			# select the highest indexes of human_probabilities such that their sum is above 1-lam
			states_to_avoid = compute_confidence_set(human_probabilities, lam)

			return self.kappa * any(s_prime in states for states in states_to_avoid)

		elif penalty_type == 'soft':
			# return self.kappa * (human_probabilities * (1 - self.distances[s_prime]/radius) * (self.distances[s_prime] <= radius)).sum()
			# lam is an effective radius.
			if lam <= 0:
				return 0
			elif lam >= min(self.rows, self.cols):
				return self.kappa
			# the two previous conditions insures that we have a saturating loss.
			return self.kappa * (human_probabilities.sum(axis=0) * np.exp(-self.distances[s_prime] / lam) * (self.distances[s_prime] <= 2*lam)).sum()
		elif penalty_type == 'ignore':
			return 0

	def _build_rewards(self, goal_state, goal_stuck=True, penalty_type='hard', lam=0.05, avoid_over=5):
		self.rewards = np.full((self.T, self.S, self.A), -np.inf)
		self.distance_rewards = np.full((self.S, self.A), -np.inf)
		self.collision_penalties = np.full((self.T, self.S, self.A), 0.)

		self.reset_occupancy_probability_human()
		for s in range(self.S):
			for a in range(self.A):
				s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
				if illegal:
					# Stay -inf
					continue
				self.distance_rewards[s, a] = self._dist_reward(s, a, s_prime, goal_state, goal_stuck=goal_stuck)
				for delta_t in range(avoid_over):
					self.collision_penalties[delta_t, s, a] = self.closeness_penalty(s_prime, t=delta_t + 1, lam=lam,
																					 penalty_type=penalty_type)
				self.rewards[:, s, a] = self.distance_rewards[s, a] - self.collision_penalties[:, s, a]

	def _update_rewards(self, t, lam=0.05, avoid_over=5, penalty_type='hard'):
		self.update_occupancy_probability_human(t)
		for s in range(self.S):
			for a in range(self.A):
				s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
				if illegal:
					# Stay -inf
					continue
				for t_p in range(t, min(self.T, t + avoid_over)):
					self.collision_penalties[t_p, s, a] = self.closeness_penalty(s_prime, t=t_p - t + 1, lam=lam,
																				 penalty_type=penalty_type)
				self.rewards[t:, s, a] = self.distance_rewards[s, a] - self.collision_penalties[t:, s, a]

	def q_values(self, goal_state, goal_stuck=True, lam=0.05, penalty_type='hard', avoid_over=5):
		"""
		compute Q values for the given goal state, with deterministic transitions
		"""
		key_tuple = (goal_state, goal_stuck, lam, penalty_type, avoid_over)
		if key_tuple in self.q_cache:
			return self.q_cache[key_tuple]
		self._build_rewards(goal_state, goal_stuck=goal_stuck, penalty_type=penalty_type, lam=lam, avoid_over=avoid_over)
		self.reward_cache[key_tuple] = np.copy(self.rewards)
		Q = np.empty([self.T, self.S, self.A])
		Q.fill(-np.inf)
		Q[-1] = self.rewards[-1]
		for t in range(self.T - 2, -1, -1):
			V = np.max(Q[t+1], axis=1)
			# Q[t] = self.rewards[t] + self.gamma * self.transition_matrix @ V
			Q[t] = self.rewards[t] + self.gamma * V[self.transitions]  # uses the fact that the transition matrix is deterministic. 100x faster than the above line.
		# for h in range(self.H-2, -1, -1):
		# 	for s in range(self.S):
		# 		for a in range(self.A):
		# 			s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
		# 			if illegal:
		# 				continue
		# 			Q[h, s, a] = self.rewards[h, s, a] + self.gamma * np.max(Q[h+1, s_prime, :])
		self.q_cache[key_tuple] = Q
		return np.copy(Q)

	def update_q_values(self, goal_state, t, goal_stuck=True, lam=0.05, prev_lam=0.05, penalty_type='hard', avoid_over=5):
		old_tuple = (goal_state, goal_stuck, prev_lam, penalty_type, avoid_over)
		new_tuple = (goal_state, goal_stuck, lam, penalty_type, avoid_over)
		assert old_tuple in self.q_cache
		# if new_tuple in self.q_cache:  # DO NOT DO THIS! The rewards have to be updated!
		# 	return self.q_cache[new_tuple]
		Q = self.q_cache[old_tuple]
		self._update_rewards(t, lam=lam, penalty_type=penalty_type, avoid_over=avoid_over)
		Q[-1] = self.rewards[-1]
		for h_p in range(self.T - 2, t - 1, -1):
			V = np.max(Q[t + 1], axis=1)
			Q[t] = self.rewards[t] + self.gamma * V[self.transitions]
			# for s in range(self.S):
			# 	for a in range(self.A):
			# 		s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
			# 		if illegal:
			# 			continue
			# 		Q[h_p, s, a] = self.rewards[h_p, s, a] + self.gamma * np.max(Q[h_p + 1, s_prime, :])
		self.q_cache[new_tuple] = Q
		return Q

	def compute_largest_uncertainty(self, t):
		"""
		Compute largest uncertainty alpha such that the state of the human at t+1 is in the confidence set of size 1- alpha
		given the information at time t.
		"""
		return compute_largest_uncertainty_level(self.list_occupancy_probability_human[t][:, 1], self.human_traj[:, t+1])

	def compute_largest_uncertainty_vector(self):
		"""
		Compute the vector of largest uncertainty alpha such that the human at t is in the confidence set of size 1- alpha[t].
		"""
		alpha_star = np.zeros(self.T-1)
		for t in range(self.T-1):
			alpha_star[t] = self.compute_largest_uncertainty(t)
		return alpha_star

	def confidence_set(self, t, p):
		"""
		Compute the confidence set of size 1-p of the position of the human at time t+1 given the information at time t.
		"""
		return compute_confidence_set(self.list_occupancy_probability_human[t][:, 1], p)
