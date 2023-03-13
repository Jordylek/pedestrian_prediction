from __future__ import division

import numpy as np
from .classic import GridWorldMDP, transition_helper, MDP2D
import pp.mdp.gridless as gridless
from pp.mdp.expanded import GridWorldExpanded, Actions, action_map


class GridWorldExpandedAdaptive(GridWorldExpanded):
	Actions = Actions

	def __init__(self, rows, cols, occupancy_probability_human, H, radius=5, penalty=50, **kwargs):
		self.H = H  # number of time steps
		self.radius = radius
		self.penalty = penalty
		# self.reward_dict = self._build_reward_dict(closeness=0.5)
		GridWorldExpanded.__init__(self, rows=rows, cols=cols, **kwargs)
		self.occupancy_probability_human = occupancy_probability_human  # matrix of size (H, S). For each time step, it contains the probability of the human being in each state
		# self.pred_traj_human = pred_traj_human  # matrix of size (H, 1). For each time step, it contains the predicted position of the human
		self.distances = self.distances_between_states()
		self.reward_cache = {}

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

	def update_occupancy_probability_human(self, occupancy_probability_human):
		self.occupancy_probability_human = occupancy_probability_human

	def _closeness_penalty(self, s_prime, radius, h):
		"""
		Compute the penalty for being too close to the human
		:param s_prime: next state
		:param radius: radius of the circle around the human
		:param penalty: penalty for being too close to the human
		:return: penalty
		"""
		human_probabilities = self.occupancy_probability_human[h]
		return self.penalty * (human_probabilities * (1 - self.distances[s_prime]/radius) * (self.distances[s_prime] <= radius)).sum()
		# return self.penalty * (human_probabilities * np.exp(- self.distances[s_prime]/radius) * (self.distances[s_prime] <= radius)).sum()

	def _build_rewards(self, goal_state, goal_stuck=True):
		self.rewards = np.full((self.H, self.S, self.A), -np.inf)
		self.distance_rewards = np.full((self.S, self.A), -np.inf)
		self.collision_penalties = np.full((self.H, self.S, self.A), 0.)

		for s in range(self.S):
			for a in range(self.A):
				s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
				if illegal:
					# Stay -inf
					continue
				self.distance_rewards[s, a] = self._dist_reward(s, a, s_prime, goal_state, goal_stuck=goal_stuck)
				for h in range(self.H):
					self.collision_penalties[h, s, a] = self._closeness_penalty(s_prime, radius=self.radius, h=h)
				self.rewards[:, s, a] = self.distance_rewards[s, a] - self.collision_penalties[:, s, a]

	def q_values(self, goal_state, goal_stuck=True, gamma=0.9):
		"""
		compute Q values for the given goal state, with deterministic transitions
		"""
		if (goal_state, goal_stuck) in self.q_cache:
			return self.q_cache[(goal_state, goal_stuck)]
		self._build_rewards(goal_state)
		self.reward_cache[(goal_state, goal_stuck)] = np.copy(self.rewards)
		Q = np.empty([self.H, self.S, self.A])
		Q.fill(-np.inf)
		Q[-1] = self.rewards[-1]
		for h in range(self.H-2, -1, -1):
			for s in range(self.S):
				for a in range(self.A):
					s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
					if illegal:
						continue
					Q[h, s, a] = self.rewards[h, s, a] + gamma * np.max(Q[h+1, s_prime, :])
		self.q_cache[(goal_state, goal_stuck)] = Q
		return np.copy(Q)




