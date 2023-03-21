from __future__ import division

import numpy as np
from .classic import GridWorldMDP, transition_helper, MDP2D
import pp.mdp.gridless as gridless
from pp.mdp.expanded import GridWorldExpanded, Actions, action_map


def compute_confidence_set(occupancy, p):
	"""
	Compute the confidence set of size 1-p for a given occupancy probability vector.
	:param occupancy: occupancy probability vector of size S
	:param p: uncertainty level.
	:return: confidence set of size 1-p
	"""
	sorted_states = np.argsort(occupancy)[::-1]
	cumulative_probs = np.cumsum(occupancy[sorted_states])
	confidence_set = sorted_states[:np.argmax(cumulative_probs >= 1 - p) + 1]
	return confidence_set


def compute_largest_uncertainty_level(occupancy, state):
	"""
	Compute the smallest size of the confidence set containing a given state.
	:param occupancy: occupancy probability vector of size S
	:param state: state
	:return: smallest size of the confidence set containing the state
	"""
	sorted_states = np.argsort(occupancy)[::-1]
	s_idx = np.where(sorted_states == state)[0][0]
	return 1 - np.sum(occupancy[sorted_states[:s_idx + 1]])

class GridWorldExpandedAdaptive(GridWorldExpanded):
	Actions = Actions

	def __init__(self, rows, cols, list_occupancy_probability_human, human_traj, H, kappa=50, gamma=0.9,
				 **kwargs):
		"""
		:param rows: number of rows of GridWorld
		:param cols: number of columns of GridWorld
		:param list_occupancy_probability_human: list of occupancy probabilities for each time step. It is a list of size H,
		and element h is a matrix of size (H-h, S).
		:param human_traj: matrix of size (H, 1). For each time step, it contains the predicted position of the human
		:param H: number of time steps
		:param kappa: parameter for the reward function
		:param gamma: discount factor
		:param kwargs:
		"""

		self.H = H  # number of time steps
		self.kappa = kappa
		self.gamma = gamma
		# self.penalty_type = penalty_type
		# assert self.penalty_type in ['hard', 'soft']
		# self.reward_dict = self._build_reward_dict(closeness=0.5)
		GridWorldExpanded.__init__(self, rows=rows, cols=cols, **kwargs)
		self.list_occupancy_probability_human = list_occupancy_probability_human
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
			for s_prime in range(self.S):
				distances[s, s_prime] = gridless.dist(self.state_to_real_coor(s), self.state_to_real_coor(s_prime))
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
	def update_occupancy_probability_human(self, h):
		self.occupancy_probability_human = self.list_occupancy_probability_human[h]

	def closeness_penalty(self, s_prime, t, lam=0.05, penalty_type='hard'):
		"""
		Compute the penalty for being at s_prime t steps from now given the current information about the human.
		:param s_prime: next state
		:param radius: radius of the circle around the human
		:param penalty: penalty for being too close to the human
		:return: penalty
		"""
		human_probabilities = self.occupancy_probability_human[t]
		assert penalty_type in ['hard', 'soft', 'ignore']
		if penalty_type == 'hard':
			# select the highest indexes of human_probabilities such that their sum is above 1-lam
			idxs = np.argsort(human_probabilities)[::-1]
			cumulative_probs = np.cumsum(human_probabilities[idxs])
			states_to_avoid = idxs[:np.argmax(cumulative_probs >= 1-lam) + 1]
			return self.kappa * (s_prime in states_to_avoid)

		elif penalty_type == 'soft':
			# return self.kappa * (human_probabilities * (1 - self.distances[s_prime]/radius) * (self.distances[s_prime] <= radius)).sum()
			# lam is an effective radius.
			if lam <= 0:
				return 0
			elif lam >= min(self.rows, self.cols):
				return self.kappa
			# the two previous conditions insures that we have a saturating loss.
			return self.kappa * (human_probabilities * np.exp(-self.distances[s_prime] / lam)).sum()
		elif penalty_type == 'ignore':
			return 0

	def _build_rewards(self, goal_state, goal_stuck=True, penalty_type='hard', lam=0.05, avoid_over=5):
		self.rewards = np.full((self.H, self.S, self.A), -np.inf)
		self.distance_rewards = np.full((self.S, self.A), -np.inf)
		self.collision_penalties = np.full((self.H, self.S, self.A), 0.)
		self.reset_occupancy_probability_human()
		for s in range(self.S):
			for a in range(self.A):
				s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
				if illegal:
					# Stay -inf
					continue
				self.distance_rewards[s, a] = self._dist_reward(s, a, s_prime, goal_state, goal_stuck=goal_stuck)
				for h in range(avoid_over):
					self.collision_penalties[h, s, a] = self.closeness_penalty(s_prime, t=h + 1, lam=lam, penalty_type=penalty_type)
				self.rewards[:, s, a] = self.distance_rewards[s, a] - self.collision_penalties[:, s, a]

	def _update_rewards(self, h, lam=0.05, avoid_over=5, penalty_type='hard'):
		self.update_occupancy_probability_human(h)
		for s in range(self.S):
			for a in range(self.A):
				s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
				if illegal:
					# Stay -inf
					continue
				for h_p in range(h, min(self.H, h + avoid_over+1)):
					self.collision_penalties[h_p, s, a] = self.closeness_penalty(s_prime, t=h_p - h + 1, lam=lam, penalty_type=penalty_type)
				self.rewards[h:, s, a] = self.distance_rewards[s, a] - self.collision_penalties[h:, s, a]

	def q_values(self, goal_state, goal_stuck=True, lam=0.05, penalty_type='hard', avoid_over=5):
		"""
		compute Q values for the given goal state, with deterministic transitions
		"""
		key_tuple = (goal_state, goal_stuck, lam, penalty_type, avoid_over)
		if key_tuple in self.q_cache:
			return self.q_cache[key_tuple]
		self._build_rewards(goal_state, goal_stuck=goal_stuck, penalty_type=penalty_type, lam=lam, avoid_over=avoid_over)
		self.reward_cache[key_tuple] = np.copy(self.rewards)
		Q = np.empty([self.H, self.S, self.A])
		Q.fill(-np.inf)
		Q[-1] = self.rewards[-1]
		for h in range(self.H-2, -1, -1):
			V = np.max(Q[h+1], axis=1)
			# Q[h] = self.rewards[h] + self.gamma * self.transition_matrix @ V
			Q[h] = self.rewards[h] + self.gamma * V[self.transitions]  # uses the fact that the transition matrix is deterministic. 100x faster than the above line.
		# for h in range(self.H-2, -1, -1):
		# 	for s in range(self.S):
		# 		for a in range(self.A):
		# 			s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
		# 			if illegal:
		# 				continue
		# 			Q[h, s, a] = self.rewards[h, s, a] + self.gamma * np.max(Q[h+1, s_prime, :])
		self.q_cache[key_tuple] = Q
		return np.copy(Q)

	def update_q_values(self, goal_state, h, goal_stuck=True, lam=0.05, prev_lam=0.05, penalty_type='hard', avoid_over=5):
		# TODO: this is a bit hacky, but it works. Make it more efficient. Especially the size of the arrays to save and how to cache them.
		old_tuple = (goal_state, goal_stuck, prev_lam, penalty_type, avoid_over)
		new_tuple = (goal_state, goal_stuck, lam, penalty_type, avoid_over)
		assert old_tuple in self.q_cache
		Q = self.q_cache[old_tuple]
		self._update_rewards(h, lam=lam, penalty_type=penalty_type, avoid_over=avoid_over)
		Q[-1] = self.rewards[-1]
		for h_p in range(self.H - 2, h-1, -1):
			V = np.max(Q[h + 1], axis=1)
			Q[h] = self.rewards[h] + self.gamma * V[self.transitions]
			# for s in range(self.S):
			# 	for a in range(self.A):
			# 		s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
			# 		if illegal:
			# 			continue
			# 		Q[h_p, s, a] = self.rewards[h_p, s, a] + self.gamma * np.max(Q[h_p + 1, s_prime, :])
		self.q_cache[new_tuple] = Q
		return Q

	def compute_largest_uncertainty(self, h):
		"""
		Compute largest uncertainty alpha such that the state of the human at h+1 is in the confidence set of size 1- alpha
		given the information at time h.
		"""
		return compute_largest_uncertainty_level(self.list_occupancy_probability_human[h][1], self.human_traj[h])

	def compute_largest_uncertainty_vector(self):
		"""
		Compute the vector of largest uncertainty alpha such that the human at h is in the confidence set of size 1- alpha[h].
		"""
		alpha_star = np.zeros(self.H)
		for h in range(self.H):
			alpha_star[h] = self.compute_largest_uncertainty(h)
		return alpha_star

	def confidence_set(self, h, p):
		"""
		Compute the confidence set of size 1-p of the position of the human at time h+1 given the information at time h.
		"""
		return compute_confidence_set(self.list_occupancy_probability_human[h][1], p)
