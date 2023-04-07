from pp.mdp import GridWorldMDP
from pp.wider_mdp.classic import GridWorldMDP
from pp.wider_mdp.adaptive_expanded import GridWorldExpandedAdaptive, compute_confidence_set
from pp.inference import hardmax as inf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import copy
from pp.learner.base_learner import BaseLearner
from pp.learner.aci_learner import ACILearner
from pp.learner.decision_learner import DecisionLearner

def plot_state_trajs(mdp, robot_trajs, human_trajs, robot_goals, human_predicted_states=None, lam=0.01, colors=None, human_color='C0',
					 plot_intervals=1, labels=None, avoid_over=None, obstacles=None, sets_to_avoid=None):

	obstacles = obstacles or []
	# preprocessing of data
	robot_coor_trajs = []
	for state_traj in robot_trajs:
		coor_traj = np.zeros((len(state_traj), 2))
		for idx in range(len(state_traj)):
			# convert from states to coordinates
			sa = state_traj[idx]
			coor = mdp.state_to_coor(sa[0])
			coor_traj[idx, :] = coor
		robot_coor_trajs.append(coor_traj)

	human_coor_trajs = []
	for state_traj in human_trajs:
		coor_traj = np.zeros((len(state_traj), 2))
		for idx in range(len(state_traj)):
			# convert from states to coordinates
			sa = state_traj[idx]
			coor = mdp.state_to_coor(sa[0])
			coor_traj[idx, :] = coor
		human_coor_trajs.append(coor_traj)

	goal_coords = np.zeros((len(robot_goals), 2))
	for idx in range(len(robot_goals)):
		goal_coords[idx, :] = mdp.state_to_coor(robot_goals[idx])

	if colors is None:
		colors = [f'C{i+1}' for i in range(len(robot_coor_trajs))]
	if labels is None:
		labels = [f'Traj {i}' for i in range(len(robot_coor_trajs))]

	rectangles = []
	for obs in obstacles:
		left = min(obs[0][0], obs[1][0])
		bottom = min(obs[0][1], obs[1][1])
		width = abs(obs[0][0] - obs[1][0])
		height = abs(obs[0][1] - obs[1][1])
		rectangles.append(plt.Rectangle((left, bottom), width, height,
							 facecolor="black", alpha=0.1))

	T = len(robot_trajs[0])
	# avoid_over is the the max avoid_over if it is an iterable, and H if it is None
	if avoid_over is None:
		avoid_over = T
	elif isinstance(avoid_over, (list, tuple)):
		avoid_over = max(avoid_over)

	assert isinstance(avoid_over, int), f'avoid_over must be an int or None, not {type(avoid_over)}'

	num_tsteps = max([coor_traj.shape[0] for coor_traj in robot_coor_trajs])
	n_plots = num_tsteps // plot_intervals
	nrows = int(np.sqrt(n_plots))
	ncols = int(np.ceil(n_plots / nrows))
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharex='all', sharey='all')

	# colors = np.linspace(0, 1, num_tsteps)
	# for tidx in range(num_tsteps-1, 0, -1):
	last_pos_size = 20
	traj_size = 1
	goal_size = 30
	goal_alpha = 0.7
	state_pred_size = 10
	obstacle_color = 'k'
	# dest_size = 80
	for i in range(nrows):
		for j in range(ncols):
			# tidx = i * ncols + j
			tidx = (i * ncols + j) * plot_intervals
			ax = axs[i, j]
			if human_predicted_states is not None:
				pred_grid = human_predicted_states[tidx]
				for t in range(pred_grid.shape[1]):
					if t >= avoid_over:
						break
					confidence_set = np.concatenate(compute_confidence_set(pred_grid[:, t], p=lam))
					confidence_coord = np.array([mdp.state_to_coor(state) for state in confidence_set])
					u = t/avoid_over
					color = mpl.colors.to_hex([0, u, 1-u])
					g = ax.scatter(confidence_coord[:, 0], confidence_coord[:, 1], s=state_pred_size * (1-u + 1e-3), marker='o',
							   alpha=0.3, c=color)
			for k, coor_traj in enumerate(robot_coor_trajs):
				color = colors[k]
				tidx1 = min(tidx, coor_traj.shape[0] - 1)
				# plot first agent's full path until timestep tidx1
				ax.plot(coor_traj[:tidx1+1, 0], coor_traj[:tidx1+1, 1], c=color, markersize=traj_size, marker='*',
						alpha=0.3, label=labels[k])
				ax.scatter(coor_traj[tidx1, 0], coor_traj[tidx1, 1], s=last_pos_size, marker='*', c=color, alpha=goal_alpha)
				ax.scatter(coor_traj[0, 0], coor_traj[0, 1], s=goal_size, marker='o', c=color, alpha=goal_alpha)  # plot start
				ax.scatter(goal_coords[k, 0], goal_coords[k, 1], s=goal_size, marker='x', c=color, alpha=goal_alpha)  # plot goal
			for k, coor_traj in enumerate(human_coor_trajs):
				color = human_color
				tidx1 = min(tidx, coor_traj.shape[0] - 1)
				# plot first agent's full path until timestep tidx1
				ax.plot(coor_traj[:tidx1+1, 0], coor_traj[:tidx1+1, 1], c=color, markersize=traj_size, marker='*',
						alpha=0.3, linestyle='--')
				ax.scatter(coor_traj[tidx1, 0], coor_traj[tidx1, 1], s=last_pos_size, marker='*', c=color, alpha=goal_alpha)
				ax.scatter(coor_traj[0, 0], coor_traj[0, 1], s=goal_size, marker='o', c=color, alpha=goal_alpha)  # plot start
				ax.scatter(coor_traj[-1, 0], coor_traj[-1, 1], s=goal_size, marker='x', c=color, alpha=goal_alpha)  # plot goal

			# Plot obstacles
			for rect in rectangles:
				ax.add_patch(rect)

			# Plot sets to avoid
			if sets_to_avoid is not None:
				if tidx >= len(sets_to_avoid):
					continue
				sets_tidx = sets_to_avoid[tidx]
				label = 'ACI prediction'
				for state in sets_tidx:
					coor = mdp.state_to_coor(state)
					rect = Rectangle((coor[0] - 0.5, coor[1] - 0.5), 1, 1, color='r', alpha=0.3, label=label)
					ax.add_patch(rect)
					label = None

			ax.set_title('T={}'.format(tidx))
			ax.set_xlim([-1, mdp.rows])
			ax.set_ylim([-1, mdp.cols])
	# add a colorbar to the figure at the bottom
	axs[-1, 0].legend(bbox_to_anchor=(-0.03, -0.03), loc="upper left", ncol=2)
	fig.show()
	return fig


def simulate(mdp, start_state, goal_state, T=None, random_traj=False, beta=1.0, goal_stuck=False):
	"""Forward simulates an optimal agent moving from start to goal.
    Args:
        mdp (MDP object): class defining the mdp model
        start_state (int): start state of agent
        goal_state (int): goal state of agent
        T (int): (optional) max path length to simulate

    Returns:
        traj (list): of (s,a) pairs that agent traversed
    """
	# goal_stuck = True  # boolean if the agent is "stuck" at the goal once they get there
	Q_value = mdp.q_values(goal_state, goal_stuck=goal_stuck)

	if random_traj:
		policy = np.exp(Q_value / beta) / np.sum(np.exp(Q_value / beta), axis=1, keepdims=True)
	else:
		policy = np.zeros_like(Q_value)
		policy[np.arange(Q_value.shape[0]), np.argmax(Q_value, axis=1)] = 1

	if T == None:
		T = np.inf

	traj = []
	s = start_state
	while len(traj) < T:
		a = np.random.choice(policy[s].shape[0], p=policy[s])
		assert a is not None
		if goal_stuck:
			if s == goal_state:
				if T < np.inf:
					traj.extend([[s, a]] * (T - len(traj)))
					break
				else:
					break
		traj.append([s, a])
		s = mdp.transition(s, a)
	return traj


def simulate_multiple_humans(mdp, H, T, start_states=None, goal_states=None, random_traj=True, betas=None, goal_stuck=False):
	if start_states is None:
		start_states = np.random.choice(mdp.S, H, replace=False)
	if goal_states is None:
		goal_states = np.random.choice(mdp.S, H, replace=False)
	if isinstance(betas, float):
		betas = np.ones(H) * betas
	elif betas is None:
		betas = np.ones(H)
	assert len(start_states) == H and len(goal_states) == H and len(betas) == H
	trajs = []
	for h in range(H):
		trajs.append(simulate(mdp, start_states[h], goal_states[h], T=T, random_traj=random_traj, beta=betas[h],
							  goal_stuck=goal_stuck))

	return start_states, goal_states, betas, trajs


def adaptive_simulation(mdp, start_state, goal_state, path_length=None, random_traj=False, beta=1.0, lam=0.5,
						penalty_type='hard'):
	"""Forward simulates an optimal agent moving from start to goal.
    Args:
        mdp (MDP object): class defining the mdp model
        start_state (int): start state of agent
        goal_state (int): goal state of agent
        path_length (int): (optional) max path length to simulate

    Returns:
        traj (list): of (s,a) pairs that agent traversed
    """

	# Get the robot's state-action value Q_r(s,a) for all states and actions, towards the specific goal goal_state
	#   Q_value_r of shape [sim_height * sim_width, num_actions]
	goal_stuck = True  # boolean if the agent is "stuck" at the goal once they get there
	Q_value = mdp.q_values(goal_state, goal_stuck=goal_stuck, lam=lam, penalty_type=penalty_type)

	# Get the action that maximizes the Q-value at each state.
	#   opt_action_r of shape [sim_height * sim_width, 1]
	if random_traj:
		policy = np.exp(Q_value / beta) / np.sum(np.exp(Q_value / beta), axis=2, keepdims=True)
		# opt_actions = np.random.choice(mdp.Actions.NUM_ACTIONS, size=mdp.num_states, p=policy)
	else:
		policy = np.zeros_like(Q_value)
		policy[np.arange(Q_value.shape[0])[:, None], np.arange(Q_value.shape[1]),  np.argmax(Q_value, axis=2)] = 1

	if path_length == None:
		path_length = np.inf

	traj = []
	s = start_state
	h = 0
	while len(traj) < path_length:
		a = np.random.choice(policy[h, s].shape[0], p=policy[h, s])
		h += 1
		# a = opt_actions[s]
		assert a is not None
		traj.append([s, a])
		# if a == mdp.Actions.ABSORB:
		# 	break
		# else:
		s = mdp.transition(s, a)
		if s == goal_state:
			if path_length < np.inf:
				traj.extend([[s, a]] * (path_length - len(traj)))
				break
	return traj


def predict_human(mdp, human_trajectory_list, dest_list, T, betas):
	"""Predicts the human recursively.
    Args:
        state_traj_h (list of list): each elevment is the list of state-action pairs that the human h actually traverses.

    Returns:
        occupancy_grids: A list of size T , where the `t`th entry is a matrix of size H x T-t x S and contains the
            probability of state s in `t` timesteps from now for human h.
    """
	occupancy_grids = []
	H = len(human_trajectory_list)
	dest_beta_prob = [None for _ in range(H)]
	for t in range(T):
		occupancy_grid = [None for _ in range(H)]
		for h in range(H):
			state_traj_h = human_trajectory_list[h]
			# if t == 0 and h ==2 :
			# 	print("debug")
			# TODO: 1 - Should not give it the information about the last action, 2- Should only be an update, not a full computation.
			occupancy_grid[h], _, d2 = inf.state.infer_joint(mdp, dest_list[h],
																		betas[h],
																		T=T - t,
																		use_gridless=False,
																		traj=state_traj_h[:t+1],
																		verbose_return=True,
																		priors=dest_beta_prob[h])
		occupancy_grid_array = np.array(occupancy_grid)
		occupancy_grids.append(occupancy_grid_array)

	return occupancy_grids


if __name__ == '__main__':
	np.random.seed(42)
	sim_height = 15  # in grid cells (not meters); treated as "rows" and corrsponds with the "x" dimension
	sim_width = 15 # in grid cells (not meters); treated as "columns" and corrsponds with the "y" dimension
	human_step_size = 3
	# MDP
	mdp = GridWorldMDP(sim_height, sim_width, max_step_size=human_step_size)
	# Setup the start and goal location for the robot (in grid cells)
	start_coor_r = [1, 1]
	goal_coor_r = [9, 9]  # [17, 5]
	start_state_r = mdp.coor_to_state(start_coor_r[0], start_coor_r[1])  # convert grid cell to "flattened" 1D state
	goal_state_r = mdp.coor_to_state(goal_coor_r[0], goal_coor_r[1])

	# Setup the start and goal location for the human (in grid cells)
	T = 25
	H = 20  # number of humans
	random_human = True
	print(f"Simulating {H} humans  trajectory...")
	if random_human:
		start_states_h = None
		goal_states_h = None
		betas_h = np.random.rand(H) * 2
	else:
		# Manually define the start and goal states for the humans
		start_coors_h = [[1, 9], [3, 3], [2, 1]]
		goal_coors_h = [[8, 2], [7, 7], [2, 9]]
		start_states_h = [mdp.coor_to_state(coor[0], coor[1]) for coor in start_coors_h]
		goal_states_h = [mdp.coor_to_state(coor[0], coor[1]) for coor in goal_coors_h]
		betas_h = [0.1, 0.8, 0.1]
	humans_start_states, humans_goal_states, true_betas, human_trajectory_list = \
		simulate_multiple_humans(mdp, H=H, T=T, start_states=start_states_h, goal_states=goal_states_h, betas=betas_h)

	# Compute the occupancy probability of each human at each time step.
	dest_list = [[goal_state] + [0] for goal_state in humans_goal_states]  # Only one potential goal for each human
	potential_betas = [[0.1, 0.5, 1, 1.5] for h in range(H)]   #
	list_occupancy_probability_human = predict_human(mdp, human_trajectory_list=human_trajectory_list, dest_list=dest_list,
									  T=T, betas=potential_betas)
	# This is a list of size T, where the `t`th entry is a matrix of size H x T-t x S and contains the probability of state s in `t` timesteps from now for human h.

	# Simulate robot: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to robot's goal
	# print("Simulating the robot's optimal trajectory [Warning: ignoring human!]...")
	# state_traj_r_ignore = simulate(mdp, start_state_r, goal_state_r, T=T)

	kappa = 100
	lam = 0.1
	human_trajs = np.array(human_trajectory_list)[:, :, 0]
	gamma = 0.9
	robot_step_size = 1
	mdp_adapt = GridWorldExpandedAdaptive(sim_height, sim_width, list_occupancy_probability_human=list_occupancy_probability_human,
										  human_traj=human_trajs, T=T, kappa=kappa, gamma=gamma,
										  max_step_size=robot_step_size)

	eta = 0.01
	alpha = 0.1
	avoid_over = 3

	print('Running adaptive simulation (HARD)...')
	hard_learner = BaseLearner(mdp=mdp_adapt, start_state=start_state_r, goal_state=goal_state_r, T=T,
							   initial_lam=lam, beta=1., penalty_type='hard', avoid_over=avoid_over)
	state_traj_r_hard = hard_learner.simulate_trajectory()

	print('Running adaptive simulation ACI...')
	aci_learner = ACILearner(mdp=mdp_adapt, start_state=start_state_r, goal_state=goal_state_r, T=T,
							 initial_lam=lam, beta=1., penalty_type='hard', eta=eta, alpha=alpha, avoid_over=avoid_over)
	state_traj_r_aci = aci_learner.simulate_trajectory()

	print('Adaptive with no collision avoidance...')
	ignore_learner = BaseLearner(mdp=mdp_adapt, start_state=start_state_r, goal_state=goal_state_r, T=T,
								 initial_lam=lam, beta=1., penalty_type='ignore', avoid_over=avoid_over)
	state_traj_r_ignore = ignore_learner.simulate_trajectory()


	# print('Running adaptive simulation (SOFT - Decision)...')
	# effective_radius = 1
	# decision_learner = DecisionLearner(mdp=mdp_adapt, start_state=start_state_r, goal_state=goal_state_r, T=T,
	# 								   initial_lam=effective_radius, beta=1., penalty_type='soft', avoid_over=avoid_over,
	# 								   eta=eta, alpha=alpha)
	# state_traj_r_decision = decision_learner.simulate_trajectory()


	print("Plotting...")
	mpl.use('Qt5Agg')
	robot_trajs = [state_traj_r_ignore, state_traj_r_hard, state_traj_r_aci]# , state_traj_r_decision]  # state_traj_r_ignore_2, state_traj_r_aci, state_traj_r_decision,
	labels = ['robot-ignore', 'robot_hard', 'robot-aci', 'robot-soft']
	robot_goals = [goal_state_r] * len(robot_trajs)
	# plot_state_trajs(mdp, robot_trajs, human_trajs, human_predicted_states=None, lam=0.01, colors=None,
	# 				 human_color='C0',
	# 				 plot_intervals=1, labels=None, avoid_over=None):
	fig = plot_state_trajs(mdp=mdp, robot_trajs=robot_trajs, robot_goals=robot_goals,
						   human_trajs=human_trajectory_list, human_predicted_states=list_occupancy_probability_human,  lam=lam,
						   avoid_over=1, labels=labels, sets_to_avoid=aci_learner.confidence_sets)
	# aci_learner.confidence_sets is a list of size T, where the `t`th entry is a list of states that the ACI learner tried to avoid when moving from t-1 to t.
	# fig.savefig('path1')
	plt.show()

# TODO:
#  1. Reward that decreases with the step size with a "hurry" parameter
#  2. Adding obstacles (see pp.mdp.car for examples)
#  3. Plot obstacles