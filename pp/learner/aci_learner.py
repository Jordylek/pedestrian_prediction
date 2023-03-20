from pp.learner.base_learner import BaseLearner
import numpy as np


class ACILearner(BaseLearner):

	def __init__(self, mdp, H, start_state, goal_state, initial_lam, beta, alpha, penalty_type='soft', eta=0.1,
				 random_traj=False, avoid_over=5):
		super().__init__(mdp, H, start_state, goal_state, initial_lam, beta, penalty_type, random_traj,
						 avoid_over=avoid_over)
		self.alpha = alpha
		self.lam_star = self.mdp.compute_largest_uncertainty_vector()
		self.eta = eta

	def update_lam(self, h, s, a):
		if h == 0:
			self.lam[0] = self.alpha
		else:
			conf_set = self.mdp.confidence_set(h, self.lam[h-1])
			error = 1 * (self.mdp.human_traj[h] not in conf_set)  # miscoverage
			self.lam[h] = self.lam[h-1] + self.eta * (self.alpha - error)
