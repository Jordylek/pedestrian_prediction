from pp.mdp import GridWorldMDP
from pp.mdp.expanded import GridWorldExpanded
from pp.mdp.adaptive_expanded import GridWorldExpandedAdaptive
from pp.inference import hardmax as inf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy


def plot_state_trajs(mdp, state_trajs, pred_states=None, dest_list=None, lamb=0.01, colors=None, radius=None,
					 plot_intervals=1, labels=None, humax_idx=0):
	# preprocessing of data
	coor_trajs = []
	for state_traj in state_trajs:
		coor_traj = np.zeros((len(state_traj), 2))
		for idx in range(len(state_traj)):
			# convert from states to coordinates
			sa = state_traj[idx]
			coor = mdp.state_to_coor(sa[0])
			coor_traj[idx, :] = coor
		coor_trajs.append(coor_traj)
	if colors is None:
		colors = [f'C{i}' for i in range(len(coor_trajs))]
	if labels is None:
		labels = [f'Traj {i}' for i in range(len(coor_trajs))]

	if pred_states is not None:
		all_likely_coor = []
		for tidx in range(pred_states.shape[0]):
			pred_grid = pred_states[tidx]
			likely_idxs = np.where(pred_grid >= lamb)  # remove all the predictions that are too unlikely
			likely_coor = []
			for state in likely_idxs[0]:
				s_coor = mdp.state_to_coor(state)
				likely_coor.append((*s_coor, pred_grid[state]))
			if len(likely_coor) > 0:
				all_likely_coor.append(np.array(likely_coor))

	num_tsteps = max([coor_traj.shape[0] for coor_traj in coor_trajs])
	n_plots = num_tsteps // plot_intervals
	nrows = int(np.sqrt(n_plots))
	ncols = int(np.ceil(n_plots / nrows))
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharex='all', sharey='all')

	# colors = np.linspace(0, 1, num_tsteps)
	# for tidx in range(num_tsteps-1, 0, -1):
	last_pos_size = 20
	traj_size = 1
	goal_size = 60
	state_pred_size = 10
	dest_size = 80
	for i in range(nrows):
		for j in range(ncols):
			# tidx = i * ncols + j
			tidx = (i * ncols + j) * plot_intervals
			ax = axs[i, j]
			# plot the predictions
			if pred_states is not None:
				tidx0 = min(tidx, len(all_likely_coor) - 1)
				coor = all_likely_coor[tidx0]
				divnorm = mpl.colors.TwoSlopeNorm(vmin=lamb, vcenter=5 * lamb, vmax=1.)
				g = ax.scatter(coor[:, 0], coor[:, 1], s=state_pred_size, marker='o', c=coor[:, 2], cmap='YlGn',
							   norm=divnorm)
				ax.set_xlim([0, mdp.rows])
				ax.set_ylim([0, mdp.cols])
			if dest_list is not None:
				for dest in dest_list:
					# convert from states to coordinates
					coor = mdp.state_to_coor(dest)
					ax.scatter(coor[0], coor[1], s=dest_size, marker='x', c='k')
			for k, coor_traj in enumerate(coor_trajs):
				color = colors[k]
				tidx1 = min(tidx, coor_traj.shape[0] - 1)
				# plot first agent's full path until timestep tidx1
				ax.plot(coor_traj[:tidx1+1, 0], coor_traj[:tidx1+1, 1], c=color, markersize=traj_size, marker='*',
						alpha=0.3, label=labels[k])
				ax.scatter(coor_traj[tidx1, 0], coor_traj[tidx1, 1], s=last_pos_size, marker='*', c=color)
				ax.scatter(coor_traj[0, 0], coor_traj[0, 1], s=goal_size, marker='o', c=color)  # plot start
				ax.scatter(coor_traj[-1, 0], coor_traj[-1, 1], s=goal_size, marker='x', c=color)  # plot goal
				if k == humax_idx:
					if radius is not None:
						# plot circle with radius radius around human agent
						circle = plt.Circle((coor_traj[tidx1, 0], coor_traj[tidx1, 1]), radius, color=color, alpha=0.2,
											fill=True)
						ax.add_artist(circle)
				# if j==0:
				# 	ax.legend()
			ax.set_title('T={}'.format(tidx))
	# add a colorbar to the figure at the bottom
	ax.legend(bbox_to_anchor=(-0.03, -0.03), loc="upper left", ncol=3)
	fig.subplots_adjust(bottom=0.2)
	cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
	fig.colorbar(g, cax=cbar_ax, orientation='horizontal')
	fig.show()
	return fig


def plot_state_trajs_prev(mdp, state_traj1, state_traj2=None, pred_states=None, dest_list=None, lamb=0.01,
					 plot_intervals=1, radius=None):
	"""Plots the traversed state trajectories.
    Args:
        mdp (MDP object):
        state_traj1 (list): of state-action pairs of agent 1
        state_traj2 (list): (optional) of state-action pairs of agent 2
        pred_states (list): (optional) of fwd_tsteps x states of agent 2
        dest_list (list): (optional) list of states that are candidate goals the human could move to
        lamb (double): parameter for thresholding likely predicted states
    """

	# preprocessing of data
	coor_traj1 = np.zeros((len(state_traj1), 2))
	for idx in range(len(state_traj1)):
		# convert from states to coordinates
		sa = state_traj1[idx]
		coor = mdp.state_to_coor(sa[0])
		coor_traj1[idx, :] = coor

	if state_traj2 is not None:
		coor_traj2 = np.zeros((len(state_traj2), 2))
		for idx in range(len(state_traj2)):
			# convert from states to coordinates
			sa = state_traj2[idx]
			coor = mdp.state_to_coor(sa[0])
			coor_traj2[idx, :] = coor

	if pred_states is not None:
		all_likely_coor = []
		for tidx in range(pred_states.shape[0]):
			pred_grid = pred_states[tidx]
			likely_idxs = np.where(pred_grid >= lamb)  # remove all the predictions that are too unlikely

			# setup a list of all the likely enough states per future timestep
			# likely_coor = None
			# for state in likely_idxs[0]:
			#     s_coor = mdp.state_to_coor(state)
			#     if likely_coor is None:
			#         likely_coor = np.array([s_coor])
			#     else:
			#         likely_coor = np.append(likely_coor, [s_coor], axis=0)
			likely_coor = []
			for state in likely_idxs[0]:
				s_coor = mdp.state_to_coor(state)
				likely_coor.append((*s_coor, pred_grid[state]))

			# import pdb; pdb.set_trace()

			if len(likely_coor) > 0:
				all_likely_coor.append(np.array(likely_coor))

	# plotting!
	# Create a figure with subplots, one per timestep in num_tsteps
	# I have num_tsteps, create a figure with num_tsteps subplots. I want it to be a square.

	# fig, axs = plt.subplots(nrow)
	# plt.figure(figsize=(int(mdp.rows*0.3),int(mdp.cols*0.3)))
	# plot the predictions
	num_tsteps = coor_traj1.shape[0]
	n_plots = num_tsteps // plot_intervals
	nrows = int(np.sqrt(n_plots))
	ncols = int(np.ceil(n_plots / nrows))
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))

	# colors = np.linspace(0, 1, num_tsteps)
	# for tidx in range(num_tsteps-1, 0, -1):
	last_pos_size = 20
	traj_size = 1
	goal_size = 60
	for i in range(nrows):
		for j in range(ncols):
			# tidx = i * ncols + j
			tidx = (i * ncols + j) * plot_intervals
			ax = axs[i, j]
			# plot the predictions
			if pred_states is not None:
				tidx0 = min(tidx, len(all_likely_coor) - 1)
				coor = all_likely_coor[tidx0]
				divnorm = mpl.colors.TwoSlopeNorm(vmin=lamb, vcenter=5 * lamb, vmax=1.)
				g = ax.scatter(coor[:, 0], coor[:, 1], s=10, marker='o', c=coor[:, 2], cmap='YlGn',
							   norm=divnorm)  # [1 - colors[tidx0], 0, colors[tidx0]])
				ax.set_xlim([0, mdp.rows])
				ax.set_ylim([0, mdp.cols])
			tidx1 = min(tidx, coor_traj1.shape[0] - 1)
			# plot first agent's full path until timestep tidx1

			ax.plot(coor_traj1[:tidx1, 0], coor_traj1[:tidx1, 1], c='g', markersize=traj_size, marker='*', alpha=0.3)
			ax.scatter(coor_traj1[tidx1, 0], coor_traj1[tidx1, 1], s=last_pos_size, marker='*', c='g')
			ax.scatter(coor_traj1[0, 0], coor_traj1[0, 1], s=goal_size, marker='o', c='g')  # plot start
			ax.scatter(coor_traj1[-1, 0], coor_traj1[-1, 1], s=goal_size, marker='x', c='g')  # plot goal

			if dest_list is not None:
				for dest in dest_list:
					# convert from states to coordinates
					coor = mdp.state_to_coor(dest)
					ax.scatter(coor[0], coor[1], s=80, marker='x', c='k')

			tidx2 = min(tidx, coor_traj2.shape[0] - 1)
			# plot second agent's full path until timestep tidx
			if state_traj2 is not None:
				ax.plot(coor_traj2[:tidx2, 0], coor_traj2[:tidx2, 1], c='r', markersize=traj_size, marker='*', alpha=0.3)
				ax.scatter(coor_traj2[tidx2, 0], coor_traj2[tidx2, 1], s=last_pos_size, marker='*', c='r')
				ax.scatter(coor_traj2[0, 0], coor_traj2[0, 1], s=goal_size, marker='o', c='r')  # plot start
				ax.scatter(coor_traj2[-1, 0], coor_traj2[-1, 1], s=goal_size, marker='x', c='r')  # plot goal
				if radius is not None:
					# plot circle with radius radius around second agent
					circle = plt.Circle((coor_traj2[tidx2, 0], coor_traj2[tidx2, 1]), radius, color='r', alpha=0.2, fill=True)
					ax.add_artist(circle)
			ax.set_title('T={}'.format(tidx))
	# add a colorbar to the figure at the bottom
	fig.subplots_adjust(bottom=0.2)
	cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
	fig.colorbar(g, cax=cbar_ax, orientation='horizontal')
	return fig


def simulate(mdp, start_state, goal_state, path_length=None, random_traj=False, beta=1.0):
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
	Q_value = mdp.q_values(goal_state, goal_stuck=goal_stuck)

	# Get the action that maximizes the Q-value at each state.
	#   opt_action_r of shape [sim_height * sim_width, 1]
	if random_traj:
		policy = np.exp(Q_value / beta) / np.sum(np.exp(Q_value / beta), axis=1, keepdims=True)
		# opt_actions = np.random.choice(mdp.Actions.NUM_ACTIONS, size=mdp.num_states, p=policy)
	else:
		policy = np.zeros_like(Q_value)
		policy[np.arange(Q_value.shape[0]), np.argmax(Q_value, axis=1)] = 1
	# opt_actions = np.argmax(Q_value, axis=1)

	if path_length == None:
		path_length = np.inf

	traj = []
	s = start_state
	while len(traj) < path_length:
		a = np.random.choice(policy[s].shape[0], p=policy[s])
		# a = opt_actions[s]
		assert a is not None
		traj.append([s, a])
		s = mdp.transition(s, a)
		if s == goal_state:
			if path_length < np.inf:
				traj.extend([[s, a]] * (path_length - len(traj)))
				break
			else:
				break
	return traj

def adaptive_simulation(mdp, start_state, goal_state, path_length=None, random_traj=False, beta=1.0):
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
	Q_value = mdp.q_values(goal_state, goal_stuck=goal_stuck)

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


def predict_human(mdp, state_traj_h, dest_list, fwd_tsteps, betas):
	"""Predicts the human recursively.
    Args:
        state_traj_h (list): of state-action pairs that the human actually traverses.

    Returns:
        occupancy_grids [np.ndarray]: A (T+1 x S) array, where the `t`th entry is the
            probability of state S in `t` timesteps from now.
    """

	# OPTION 1: The line below feeds in the entire human traj history so far
	# 			and does a single bulk Bayesian inference step.

	# Here, Traj is only use for the INITIAL step and for the trajectory length.
	straj = copy.deepcopy(state_traj_h)
	straj.reverse()
	occupancy_grids, beta_occu, dest_beta_prob = inf.state.infer_joint(mdp,
																	   dest_list,
																	   betas,
																	   T=fwd_tsteps,
																	   use_gridless=False,
																	   traj=straj,
																	   verbose_return=True)

	# OPTION 2: The line below feeds in the last human (s,a) pair and previous posterior
	# 			and does a recursive Bayesian update.
	# occupancy_grids, beta_occu, dest_beta_prob = inf.state.infer_joint(mdp,
	#                                                                      dest_list,
	#                                                                      betas,
	#                                                                      T=fwd_tsteps,
	#                                                                      use_gridless=True,
	#                                                                      priors=dest_beta_prob,
	#                                                                      traj=straj[-2:],
	#                                                                      verbose_return=True)

	return occupancy_grids


if __name__ == '__main__':
	sim_height = 20  # in grid cells (not meters); treated as "rows" and corrsponds with the "x" dimension
	sim_width = 26  # in grid cells (not meters); treated as "columns" and corrsponds with the "y" dimension
	# MDP
	mdp = GridWorldExpanded(sim_height, sim_width)
	# Setup the start and goal location for the robot (in grid cells)
	start_coor_r = [2, 2]
	goal_coor_r = [2, 18]  # [17, 5]
	start_state_r = mdp.coor_to_state(start_coor_r[0], start_coor_r[1])  # convert grid cell to "flattened" 1D state
	goal_state_r = mdp.coor_to_state(goal_coor_r[0], goal_coor_r[1])

	# Setup start and possible goals for the human (in grid cells)
	start_coor_h = [2, 10]
	goal_coor_h = [[2, 5], [6, 10]]  # [[2, 3], [10, 25]] # hypothesis space of possible goals the human has
	start_state_h = mdp.coor_to_state(start_coor_h[0], start_coor_h[1])  # convert grid cell to "flattened" 1D state
	goal_state_h = [mdp.coor_to_state(g[0], g[1]) for g in goal_coor_h]

	# Setup the simulated humans true goal they are moving to.
	true_goal_idx = 0
	true_goal_coor_h = goal_coor_h[true_goal_idx]
	true_goal_state_h = goal_state_h[true_goal_idx]
	H = 30

	# Simulate robot: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to robot's goal
	print("Simulating the robot's optimal trajectory [Warning: ignoring human!]...")
	state_traj_r_ignore = simulate(mdp, start_state_r, goal_state_r, path_length=H)

	# Simulate human: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to true human goal
	print("Simulating the human's optimal trajectory...")
	state_traj_h = simulate(mdp, start_state_h, true_goal_state_h, random_traj=True,
							beta=0.5, path_length=H)  # beta=0.1 is the human's rationality. Smaller is better

	# Predict the human
	fwd_tsteps = len(state_traj_h)
	betas = [0.1, 1, 10]  # assume the human is rational when predicting them. beta here is 1/beta in the paper.

	print("Predicting human...")
	# H = len(state_traj_h)
	pred_state_traj_h = predict_human(mdp, state_traj_h, goal_state_h, H, betas)
	radius = 3
	penalty = 20
	mdp_adapt = GridWorldExpandedAdaptive(sim_height, sim_width, occupancy_probability_human=pred_state_traj_h, H=H,
										  radius=radius, penalty=penalty)
	mdp_adapt.q_values(goal_state_r, goal_stuck=True)

	state_traj_r = adaptive_simulation(mdp_adapt, start_state_r, goal_state_r, path_length=H)
	lamb = 0.01
	print("Plotting...")
	mpl.use('Qt5Agg')
	fig = plot_state_trajs(mdp, [state_traj_h, state_traj_r_ignore, state_traj_r], pred_state_traj_h, goal_state_h, lamb=lamb,
					 radius=radius, humax_idx=0,
					 labels=['human', 'robot-ignore', 'robot-adapt'])
	fig.savefig('path1')
	plt.show()
	# TODO:
	#  1. Add online update of the human's occupancy predictions
	#    1.a Using position coverage
	#    1.b Using decision coverage


	# # Setup the GridWorld
	# sim_height = 20  # in grid cells (not meters); treated as "rows" and corrsponds with the "x" dimension
	# sim_width = 26  # in grid cells (not meters); treated as "columns" and corrsponds with the "y" dimension
	# mdp = GridWorldExpanded(sim_height, sim_width)
	#
	# # Setup the start and goal location for the robot (in grid cells)
	# start_coor_r = [2, 2]
	# goal_coor_r =  [2, 18]  # [17, 5]
	# start_state_r = mdp.coor_to_state(start_coor_r[0], start_coor_r[1])  # convert grid cell to "flattened" 1D state
	# goal_state_r = mdp.coor_to_state(goal_coor_r[0], goal_coor_r[1])
	#
	# # Setup start and possible goals for the human (in grid cells)
	# start_coor_h = [2, 10]
	# goal_coor_h = [[2, 10], [2, 6]]  # [[2, 3], [10, 25]] # hypothesis space of possible goals the human has
	# start_state_h = mdp.coor_to_state(start_coor_h[0], start_coor_h[1])  # convert grid cell to "flattened" 1D state
	# goal_state_h = [mdp.coor_to_state(g[0], g[1]) for g in goal_coor_h]
	#
	# # Setup the simulated humans true goal they are moving to.
	# true_goal_idx = 1
	# true_goal_coor_h = goal_coor_h[true_goal_idx]
	# true_goal_state_h = goal_state_h[true_goal_idx]
	#
	# # Simulate robot: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to robot's goal
	# # print("Simulating the robot's optimal trajectory [Warning: ignoring human!]...")
	# # state_traj_r = simulate(mdp, start_state_r, goal_state_r)
	#
	# # Simulate human: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to true human goal
	# print("Simulating the human's optimal trajectory...")
	# H = 20
	# state_traj_h = simulate(mdp, start_state_h, true_goal_state_h, random_traj=True,
	# 						beta=0.5, path_length=H)  # beta=0.1 is the human's rationality. Smaller is better
	#
	# # Predict the human
	# fwd_tsteps = len(state_traj_h)
	# betas = [0.1, 1, 10]  # assume the human is rational when predicting them. beta here is 1/beta in the paper.
	# print("Predicting human...")
	# # H = len(state_traj_h)
	# pred_state_traj_h = predict_human(mdp, state_traj_h, goal_state_h, H, betas)
	# radius = 0
	# penalty = 0
	# mdp_adapt = GridWorldExpandedAdaptive(sim_height, sim_width, occupancy_probability_human=pred_state_traj_h, H=H,
	# 									  radius=radius, penalty=penalty)
	# mdp_adapt.q_values(goal_state_r, goal_stuck=True)
	# state_traj_r = adaptive_simulation(mdp_adapt, start_state_r, goal_state_r, path_length=H)
	#
	# # import matplotlib as mpl
	# # mpl.use('TkAgg')
	# mpl.use('Qt5Agg')
	# # Plot the optimal path
	# lamb = 0.01
	# print("Plotting...")
	# fig = plot_state_trajs(mdp, state_traj_r, state_traj_h, pred_state_traj_h, goal_state_h, lamb=lamb, radius=radius)

