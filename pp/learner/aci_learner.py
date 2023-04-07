from pp.learner.base_learner import BaseLearner
import numpy as np


class ACILearner(BaseLearner):

	def __init__(self, mdp, T, start_state, goal_state, initial_lam, beta, alpha, penalty_type='soft', eta=0.1,
				 goal_stuck=True, random_traj=False, avoid_over=5):
		super().__init__(mdp, T, start_state, goal_state, initial_lam, beta, penalty_type, goal_stuck=goal_stuck,
						 random_traj=random_traj, avoid_over=avoid_over)
		self.alpha = alpha
		self.lam_star = self.mdp.compute_largest_uncertainty_vector()
		self.eta = eta
		self.confidence_sets = []

	def update_lam(self, t, s, a):
		if t == 0:
			self.lam[0] = self.alpha
			self.confidence_sets.append(np.array([]))
		else:
			conf_set_per_human = self.mdp.confidence_set(t, self.lam[t-1])
			self.confidence_sets.append(np.sort(np.unique(np.concatenate(conf_set_per_human))))
			H = self.mdp.H
			error = 1 * any(self.mdp.human_traj[h, t] not in conf_set_per_human[h] for h in range(H))
			self.lam[t] = self.lam[t-1] + self.eta * (self.alpha - error)


