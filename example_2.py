from pp.mdp import GridWorldMDP
from pp.mdp.expanded import GridWorldExpanded
from pp.inference import hardmax as inf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

def plot_state_trajs(mdp, state_traj1, state_traj2=None, pred_states=None, dest_list=None, lamb=0.01,
                     plot_intervals=1):
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
            likely_idxs = np.where(pred_grid >= lamb) # remove all the predictions that are too unlikely

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
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * nrows, 3 * ncols))

    # colors = np.linspace(0, 1, num_tsteps)
    # for tidx in range(num_tsteps-1, 0, -1):
    for i in range(nrows):
        for j in range(ncols):
            # tidx = i * ncols + j
            tidx = (i * ncols + j) * plot_intervals
            ax = axs[i, j]
            # plot the predictions
            if pred_states is not None:
                tidx0 = min(tidx, len(all_likely_coor) - 1)
                coor = all_likely_coor[tidx0]
                divnorm = mpl.colors.TwoSlopeNorm(vmin=lamb, vcenter=5*lamb, vmax=1.)
                g = ax.scatter(coor[:, 0], coor[:, 1], s=20, marker='o', c= coor[:, 2], cmap='YlGn', norm=divnorm)  # [1 - colors[tidx0], 0, colors[tidx0]])
                ax.set_xlim([0, mdp.rows])
                ax.set_ylim([0, mdp.cols])
            tidx1 = min(tidx, coor_traj1.shape[0] - 1)
            # plot first agent's full path until timestep tidx1

            ax.plot(coor_traj1[:tidx1,0], coor_traj1[:tidx1,1], '-g', marker='*')
            ax.scatter(coor_traj1[0, 0], coor_traj1[0, 1], s=60, marker='o', c='g')  # plot start
            ax.scatter(coor_traj1[-1, 0], coor_traj1[-1, 1], s=60, marker='x', c='g') # plot goal

            if dest_list is not None:
                for dest in dest_list:
                    # convert from states to coordinates
                    coor = mdp.state_to_coor(dest)
                    ax.scatter(coor[0], coor[1], s=80, marker='x', c='k')

            tidx2 = min(tidx, coor_traj2.shape[0] - 1)
            # plot second agent's full path until timestep tidx
            if state_traj2 is not None:
                ax.plot(coor_traj2[:tidx2,0], coor_traj2[:tidx2,1], '-r', marker='*')
                ax.scatter(coor_traj2[0, 0], coor_traj2[0, 1], s=60, marker='o', c='r')  # plot start
                ax.scatter(coor_traj2[-1, 0], coor_traj2[-1, 1], s=60, marker='x', c='r') # plot goal
            ax.set_title('T={}'.format(tidx))
    # add a colorbar to the figure at the bottom
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    fig.colorbar(g, cax=cbar_ax, orientation='horizontal')

    plt.show()
            #

            # coor = all_likely_coor[tidx]
            # ax = axs[i, j]
            # ax.scatter(coor[:, 0], coor[:, 1], s=20, marker='o', c=[1 - colors[tidx], 0, colors[tidx]])
            # ax.set_xlim([0, mdp.rows])
            # ax.set_ylim([0, mdp.cols])
    # if pred_states is not None:



    # # plot first agent's full path
    # plt.plot(coor_traj1[:,0], coor_traj1[:,1], '-g', marker='*')
    # plt.scatter(coor_traj1[0, 0], coor_traj1[0, 1], s=60, marker='o', c='g')  # plot start
    # plt.scatter(coor_traj1[-1, 0], coor_traj1[-1, 1], s=60, marker='x', c='g') # plot goal
    #
    #
    # # plot candidate goals for second agent, if we have them
    # if dest_list is not None:
    #     for dest in dest_list:
    #         # convert from states to coordinates
    #         coor = mdp.state_to_coor(dest)
    #         plt.scatter(coor[0], coor[1], s=80, marker='x', c='k')  # plot goal
    #
    # # plot second agent's full path, if we have it
    # if state_traj2 is not None:
    #     plt.plot(coor_traj2[:, 0], coor_traj2[:, 1], '--r')
    #     plt.scatter(coor_traj2[0, 0], coor_traj2[0, 1], s=60, marker='o', c='r')  # plot start
    #     plt.scatter(coor_traj2[-1, 0], coor_traj2[-1, 1], s=60, marker='x', c='r')  # plot goal
    #
    #
    # # setup bounds of env
    # plt.xlim([0, mdp.rows])
    # plt.ylim([0, mdp.cols])


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
        policy = np.exp(Q_value/beta)/np.sum(np.exp(Q_value/beta), axis=1, keepdims=True)
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
        if a == mdp.Actions.ABSORB:
            break
        else:
            s = mdp.transition(s, a)
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
    # Setup the GridWorld
    sim_height = 20 # in grid cells (not meters); treated as "rows" and corrsponds with the "x" dimension
    sim_width = 26 # in grid cells (not meters); treated as "columns" and corrsponds with the "y" dimension
    mdp = GridWorldExpanded(sim_height, sim_width)

    # Setup the start and goal location for the robot (in grid cells)
    start_coor_r = [1, 15]
    goal_coor_r = [17, 5]
    start_state_r = mdp.coor_to_state(start_coor_r[0], start_coor_r[1]) # convert grid cell to "flattened" 1D state
    goal_state_r = mdp.coor_to_state(goal_coor_r[0], goal_coor_r[1])

    # Setup start and possible goals for the human (in grid cells)
    start_coor_h = [17, 17]
    goal_coor_h =  [[2, 3], [10, 3]]# [[2, 3], [10, 25]] # hypothesis space of possible goals the human has
    start_state_h = mdp.coor_to_state(start_coor_h[0], start_coor_h[1]) # convert grid cell to "flattened" 1D state
    goal_state_h = [mdp.coor_to_state(g[0], g[1]) for g in goal_coor_h]

    # Setup the simulated humans true goal they are moving to.
    true_goal_idx = 1
    true_goal_coor_h = goal_coor_h[true_goal_idx]
    true_goal_state_h = goal_state_h[true_goal_idx]

    # Simulate robot: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to robot's goal
    print("Simulating the robot's optimal trajectory [Warning: ignoring human!]...")
    state_traj_r = simulate(mdp, start_state_r, goal_state_r)
    # TODO: Right now the robot doesn't avoid collision with the human!

    # Simulate human: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to true human goal
    print("Simulating the human's optimal trajectory...")
    state_traj_h = simulate(mdp, start_state_h, true_goal_state_h, random_traj=True, beta=0.5)  # beta=0.1 is the human's rationality. Smaller is better

    # Predict the human
    fwd_tsteps = len(state_traj_h)
    betas = [0.1, 1, 10]  # assume the human is rational when predicting them. beta here is 1/beta in the paper.
    print("Predicting human...")
    pred_state_traj_h = predict_human(mdp, state_traj_h, goal_state_h, fwd_tsteps, betas)
    # import matplotlib as mpl
    # mpl.use('TkAgg')
    mpl.use('Qt5Agg')
    # Plot the optimal path
    lamb = 0.01
    print("Plotting...")
    plot_state_trajs(mdp, state_traj_r, state_traj_h, pred_state_traj_h, goal_state_h, lamb=lamb)
    # TODO: Show states in a sequence of timestamps, not just a single timestamp. DONE
    # TODO:
    #  Change the reward to include non-collision with the human (or where the human is predicted to be).
    #  Try a full planning offline at t=0 (probably not feasible)
    #  Try to update online with the human's actual trajectory
    #  Toy case: robot wants to go from (0,0) to (0,1) and human stands on (0, 0.5). Check if robot goes around
    #  Example of reward function R = -dist(next_pos, goal) + dist(next_pos, human) * (if ||next_pos - human|| < r)
    #    and r is a hyperparameter.

