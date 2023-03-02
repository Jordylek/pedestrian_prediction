from pp.mdp import GridWorldMDP
from pp.mdp.expanded import GridWorldExpanded
from pp.inference import hardmax as inf
import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_state_trajs(mdp, state_traj1, state_traj2=None, pred_states=None, dest_list=None, lamb=0.01):
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
            likely_coor = None
            for state in likely_idxs[0]:
                s_coor = mdp.state_to_coor(state)
                if likely_coor is None:
                    likely_coor = np.array([s_coor])
                else:
                    likely_coor = np.append(likely_coor, [s_coor], axis=0)

            # import pdb; pdb.set_trace()

            if len(likely_coor) > 0:
                all_likely_coor.append(likely_coor)

    # plotting!
    plt.figure(figsize=(int(mdp.rows*0.3),int(mdp.cols*0.3)))

    # plot the predictions
    if pred_states is not None:
        num_tsteps = pred_states.shape[0]
        colors = np.linspace(0, 1, pred_states.shape[0])
        for tidx in range(pred_states.shape[0]-1, 0, -1):
            coor = all_likely_coor[tidx]
            plt.scatter(coor[:, 0], coor[:, 1], s=20, marker='o', c=[1-colors[tidx], 0, colors[tidx]])

    # plot first agent's full path
    plt.plot(coor_traj1[:,0], coor_traj1[:,1], '-g')
    plt.scatter(coor_traj1[0, 0], coor_traj1[0, 1], s=60, marker='o', c='g')  # plot start
    plt.scatter(coor_traj1[-1, 0], coor_traj1[-1, 1], s=60, marker='x', c='g') # plot goal


    # plot candidate goals for second agent, if we have them
    if dest_list is not None:
        for dest in dest_list:
            # convert from states to coordinates
            coor = mdp.state_to_coor(dest)
            plt.scatter(coor[0], coor[1], s=80, marker='x', c='k')  # plot goal

    # plot second agent's full path, if we have it
    if state_traj2 is not None:
        plt.plot(coor_traj2[:, 0], coor_traj2[:, 1], '--r')
        plt.scatter(coor_traj2[0, 0], coor_traj2[0, 1], s=60, marker='o', c='r')  # plot start
        plt.scatter(coor_traj2[-1, 0], coor_traj2[-1, 1], s=60, marker='x', c='r')  # plot goal


    # setup bounds of env
    plt.xlim([0, mdp.rows])
    plt.ylim([0, mdp.cols])
    plt.show()

def simulate(mdp, start_state, goal_state, path_length=None):
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
    opt_actions = np.argmax(Q_value, axis=1)

    if path_length == None:
        path_length = np.inf

    traj = []
    s = start_state
    while len(traj) < path_length:
        a = opt_actions[s]
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
    # (self.occupancy_grids, self.beta_occu, self.dest_beta_prob) = inf.state.infer_joint(self.gridworld,
    #                                                                                     dest_list,
    #                                                                                     self.betas,
    #                                                                                     T=self.fwd_tsteps,
    #                                                                                     use_gridless=True,
    #                                                                                     priors=self.dest_beta_prob,
    #                                                                                     traj=traj[-2:],
    #                                                                                     epsilon_dest=self.epsilon_dest,
    #                                                                                     epsilon_beta=self.epsilon_beta,
    #                                                                                     verbose_return=True)

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
    goal_coor_h = [[2, 3], [10, 25]] # hypothesis space of possible goals the human has
    start_state_h = mdp.coor_to_state(start_coor_h[0], start_coor_h[1]) # convert grid cell to "flattened" 1D state
    goal_state_h = [mdp.coor_to_state(g[0], g[1]) for g in goal_coor_h]

    # Setup the simulated humans true goal they are moving to.
    true_goal_idx = 0
    true_goal_coor_h = goal_coor_h[true_goal_idx]
    true_goal_state_h = goal_state_h[true_goal_idx]

    # Simulate robot: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to robot's goal
    print("Simulating the robot's optimal trajectory [Warning: ignoring human!]...")
    state_traj_r = simulate(mdp, start_state_r, goal_state_r)
    # TODO: Right now the robot doesn't avoid collision with the human!

    # Simulate human: returns optimal [(s,a)_0, ..., (s,a)_T] trajectory from start to true human goal
    print("Simulating the human's optimal trajectory...")
    state_traj_h = simulate(mdp, start_state_h, true_goal_state_h)

    # Predict the human
    fwd_tsteps = 10
    betas = [0.1]  # assume the human is rational when predicting them
    print("Predicting human...")
    pred_state_traj_h = predict_human(mdp, state_traj_h, goal_state_h, fwd_tsteps, betas)

    # Plot the optimal path
    lamb = 0.000001
    print("Plotting...")
    plot_state_trajs(mdp, state_traj_r, state_traj_h, pred_state_traj_h, goal_state_h, lamb=lamb)
