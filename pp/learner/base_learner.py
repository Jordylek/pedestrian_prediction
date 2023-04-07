import numpy as np


class BaseLearner:
	def __init__(self, mdp, T, start_state, goal_state, initial_lam, beta, penalty_type='soft', goal_stuck=True,
				 random_traj=False, avoid_over=5):
		self.mdp = mdp
		self.T = T
		self.start_state = start_state
		self.goal_state = goal_state
		self.initial_lam = initial_lam
		self.lam = np.full(self.T, 0.)
		self.beta = beta
		self.policy = np.zeros((self.T, self.mdp.S, self.mdp.A))
		self.random_traj = random_traj
		self.penalty_type = penalty_type
		self.goal_stuck = goal_stuck
		self.avoid_over = avoid_over

	def update_policy(self, t, s, a):
		self.update_lam(t, s, a)
		if t == 0:
			self._set_q_values()
		if t > 0:
			if self.lam[t] == self.lam[t-1] and self.penalty_type == 'ignore':
				return self.policy
			self.update_q_value(t)
		Q_value = self.Q_value[t:]
		if self.random_traj:
			self.policy[t:] = np.exp(Q_value / self.beta) / np.sum(np.exp(Q_value / self.beta), axis=1, keepdims=True)
			# opt_actions = np.random.choice(mdp.Actions.NUM_ACTIONS, size=mdp.num_states, p=policy)
		else:
			self.policy[t:] = np.eye(self.mdp.A)[np.argmax(Q_value, axis=2)]
			# self.policy[np.arange(t, self.t)[:, None], np.arange(self.mdp.S),  np.argmax(Q_value, axis=2)] = 1
		return self.policy

	def _set_q_values(self):
		self.Q_value = self.mdp.q_values(self.goal_state, goal_stuck=self.goal_stuck, lam=self.lam[0],
										 penalty_type=self.penalty_type, avoid_over=self.avoid_over)

	def update_q_value(self, t):
		# Update
		self.Q_value =self.mdp.update_q_values(self.goal_state, t, goal_stuck=self.goal_stuck, lam=self.lam[t],
											   prev_lam=self.lam[t-1], penalty_type=self.penalty_type, avoid_over=self.avoid_over)

	def update_lam(self, t, s, a):
		self.lam[t] = self.initial_lam

	def simulate_trajectory(self):
		traj = []
		s = self.start_state
		A = self.mdp.A
		t = 0
		a = 0
		while len(traj) < self.T:
			policy = self.update_policy(t, s, a)[t]
			a = np.random.choice(A, p=policy[s])
			t += 1
			assert a is not None
			traj.append([s, a])
			s = self.mdp.transition(s, a)
			if s == self.goal_state:
				if self.T < np.inf:
					traj.extend([[s, a]] * (self.T - len(traj)))
					break
		return traj

