from pp.learner.base_learner import BaseLearner
import numpy as np


class DecisionLearner(BaseLearner):

	def __init__(self, mdp, T, start_state, goal_state, initial_lam, beta, alpha, penalty_type='soft', eta=0.1,
				 goal_stuck=True, random_traj=False, avoid_over=5):
		super().__init__(mdp, T, start_state, goal_state, initial_lam, beta, penalty_type, goal_stuck=goal_stuck,
						 random_traj=random_traj, avoid_over=avoid_over)
		self.alpha = alpha
		self.eta = eta

	def update_lam(self, t, s, a):
		if t == 0:
			self.lam[0] = self.initial_lam
		else:
			prev_lam = self.lam[t-1]
			assert a is not None
			if self.penalty_type == 'hard':
				# loss is 1 if we miscover
				conf_set = self.mdp.confidence_set(t, prev_lam)
				loss = 1 * any(self.mdp.human_traj[:, t] not in conf_set)  # miscoverage
			elif self.penalty_type == 'soft':
				# loss is relative to the distance to the human
				if prev_lam <= 0:
					loss = 0
				elif prev_lam >= min(self.mdp.rows, self.mdp.cols):
					loss = 1
				else:
					s_prime = self.mdp.transition(s, a)
					loss = (np.exp(-self.mdp.distances[s_prime, self.mdp.human_traj[:, t]] / prev_lam) * (self.mdp.distances[s_prime, self.mdp.human_traj[:, t]] <= 2*prev_lam)).sum()
			elif self.penalty_type == 'ignore':
				loss = 0
			else:
				raise NotImplementedError
			self.lam[t] = self.lam[t-1] + self.eta * (self.alpha - loss)
