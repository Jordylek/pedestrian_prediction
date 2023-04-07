from __future__ import division

from enum import IntEnum
import numpy as np
from .hardmax import forwards_value_iter as _value_iter
from .mdp import MDP


class Directions(IntEnum):
    ABSORB = 0
    UP = 1
    UP_RIGHT = 2
    RIGHT = 3
    DOWN_RIGHT = 4
    DOWN = 5
    DOWN_LEFT = 6
    LEFT = 7
    UP_LEFT = 8

    @staticmethod
    def get_directions():
        return {Directions.UP, Directions.UP_RIGHT, Directions.RIGHT, Directions.DOWN_RIGHT,
                Directions.DOWN, Directions.DOWN_LEFT, Directions.LEFT, Directions.UP_LEFT}

    @staticmethod
    def get_diagonal_directions():
        return {Directions.UP_LEFT, Directions.UP_RIGHT, Directions.DOWN_LEFT,
                Directions.DOWN_RIGHT}

    @staticmethod
    def get_cardinal_directions():
        return {Directions.UP, Directions.RIGHT, Directions.DOWN, Directions.LEFT}


class ActionConverter(object):

    def __init__(self, max_step_size=3):
        self.max_step_size = max_step_size

    def __len__(self):
        return 1 + 8 * self.max_step_size

    def __getitem__(self, key):
        assert key >= 0 and key < len(self), key
        if key == 0:
            return 0, Directions.ABSORB
        key -= 1
        step_size = key // 8 + 1
        direction = Directions(key % 8 + 1)
        return step_size, direction


diagonal_actions = {Directions.UP_LEFT, Directions.UP_RIGHT, Directions.DOWN_LEFT,
                    Directions.DOWN_RIGHT}

# XXX: optimize so that we don't need to convert between state and coor.
def transition_helper(g, s, a, alert_illegal=False, action_converter=None):

    assert action_converter is not None
    r, c = g.state_to_coor(s)
    assert a >= 0, a
    step_size, direction = action_converter[a]
    r_prime, c_prime = r, c
    if direction == Directions.LEFT:
        r_prime = r - step_size
    elif direction == Directions.RIGHT:
        r_prime = r + step_size
    elif direction == Directions.DOWN:
        c_prime = c - step_size
    elif direction == Directions.UP:
        c_prime = c + step_size
    elif direction == Directions.UP_LEFT:
        r_prime, c_prime = r - step_size, c + step_size
    elif direction == Directions.UP_RIGHT:
        r_prime, c_prime = r + step_size, c + step_size
    elif direction == Directions.DOWN_LEFT:
        r_prime, c_prime = r - step_size, c - step_size
    elif direction == Directions.DOWN_RIGHT:
        r_prime, c_prime = r + step_size, c - step_size
    elif direction == Directions.ABSORB:
        pass
    else:
        raise BaseException("undefined action {}".format(direction))

    illegal = False
    if r_prime < 0 or r_prime >= g.rows or \
            c_prime < 0 or c_prime >= g.cols:
        r_prime, c_prime = r, c
        illegal = True

    s_prime = g.coor_to_state(r_prime, c_prime)

    if alert_illegal:
        return s_prime, illegal
    else:
        return s_prime


# TODO: rename this class into GridWorldMDP and rename GridWorldMDP to
#  GridWorldMDPClassic.

class MDP2D(MDP):
    def __init__(self, rows, cols, A, reward_dict={}, **kwargs):
        """
        Superclass for GridWorldMDP and GridWorldExpanded.

        Params:
            rows [int]: The number of rows.
            cols [int]: The number of columns.
            A [int]: The number of actions.

        Debug Params (mainly used in unittests):
            reward_dict [dict]: Maps state `s_new` to reward `R`. Passing in a
                nonempty dict for this parameter will make any legal
                state-action pair that transitions to `s_new` yield the reward
                `R`.
        """
        assert rows > 0
        assert cols > 0
        assert isinstance(rows, int)
        assert isinstance(cols, int)

        # TODO: Rename rows=> X and rename cols=> Y. The current naming
        # convention is confusing.
        self.rows = rows
        self.cols = cols
        S = rows * cols

        # Convert from coordinates to state number as required by super-class
        reward_dict = {self.coor_to_state(x, y): R
                for (x, y), R in reward_dict.items()}

        MDP.__init__(self, S=S, A=A, reward_dict=reward_dict, **kwargs)
        self.q_cache = {}


    def coor_to_state(self, r, c):
        """
        Params:
            r [int]: The state's row.
            c [int]: The state's column.

        Returns:
            s [int]: The state number associated with the given coordinates in
                a standard grid world.
        """
        assert 0 <= r < self.rows, "invalid (rows, r)={}".format((self.rows, r))
        assert 0 <= c < self.cols, "invalid (cols, c)={}".format((self.cols, c))
        return r * self.cols + c

    def state_to_coor(self, s):
        """
        Params:
            s [int]: The state.

        Returns:
            r, c [int]: The row and column associated with state s.
        """
        assert s < self.rows * self.cols
        return s // self.cols, s % self.cols

    def state_to_real_coor(self, s):
        x, y = self.state_to_coor(s)
        return np.array([x + 0.5, y + 0.5])


# Classic Gridworld
class GridWorldMDP(MDP2D):
    Directions = Directions

    def __init__(self, rows, cols, max_step_size=3, goal_state=None, step_cost=0.5,
            allow_wait=True, obstacles_list=None, **kwargs):
        """
        An agent in a GridWorldMDP can move between adjacent/diagonal cells.

        If the agent chooses an illegal action it receives a float('-inf')
        reward and will stay in place.

        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            goal_state [int]: (optional) The goal state at which ABSORB is legal
                and costs 0.
            euclidean_rewards [bool]: (optional) If True, then scale rewards for
                moving diagonally by sqrt(2).
            allow_wait [bool]: (optional) If False, then the ABSORB action is
                illegal in all states except the goal. If True, then the ABSORB
                action costs default_reward in states other than the goal.
            obstacles_list [list]: (optional) A list of obstacles. Each obstacle is a tuple of tuples ((x1, y1), (x2, y2)) where
            (x1, y1) is the top left corner and (x2, y2) is the bottom right corner of the obstacle.
            step_cost [float]: (optional) The cost of taking a step.
        """
        if goal_state is not None:
            assert isinstance(goal_state, int)

        self.allow_wait = allow_wait
        self.max_step_size = max_step_size
        self.action_converter = ActionConverter(max_step_size=self.max_step_size)
        self.step_cost = step_cost
        self.obstacles_list = [] if obstacles_list is None else obstacles_list
        self._valid_states = None
        self._valid_transitions = None
        A = len(self.action_converter)
        MDP2D.__init__(self, rows=rows, cols=cols, A=A,
                       transition_helper=self._transition_helper, **kwargs)


        if self.allow_wait:
            self.rewards[:, Directions.ABSORB].fill(self.default_reward)
        else:
            self.rewards[:, Directions.ABSORB].fill(-np.inf)

    def _compute_valid_states(self):
        if self.obstacles_list is None:
            return range(self.S)

        valid_states = np.full(self.S, True)
        for s in range(self.S):
            x, y = self.state_to_coor(s)
            for obstacle in self.obstacles_list:
                (x1, y1), (x2, y2) = obstacle
                if x1 < x < x2 and y1 < y < y2:
                    valid_states[s] = False
                    break
        self._valid_states = valid_states

    def _compute_valid_transitions(self):
        valid_transitions = np.full((self.S, self.S), True)

        for s in range(self.S):
            x, y = self.state_to_coor(s)
            if not self.valid_states[s]:
                valid_transitions[s, :] = False
                valid_transitions[:, s] = False
            for s_prime in range(s, self.S):
                x_prime, y_prime = self.state_to_coor(s_prime)
                for obstacle in self.obstacles_list:
                    illegal = is_valid(x, y, x_prime, y_prime, obstacle)
                    if illegal:
                        valid_transitions[s, s_prime] = False
                        valid_transitions[s_prime, s] = False
                        break
        self._valid_transitions = valid_transitions

    @property
    def valid_states(self):
        if self._valid_states is None:
            self._compute_valid_states()
            self._compute_valid_transitions()
        return self._valid_states

    @property
    def valid_transitions(self):
        if self._valid_transitions is None:
            self._compute_valid_states()
            self._compute_valid_transitions()
        return self._valid_transitions
    # XXX: optimize so that we don't need to convert between state and coor.
    def _transition_helper(self, s, a, alert_illegal=False):
        if self.valid_states is None:
            # Compute the valid states and transitions
            self._valid_states()
            self._valid_transitions()
        out = transition_helper(self, s, a, alert_illegal=alert_illegal, action_converter=self.action_converter)
        if alert_illegal:
            s_prime, illegal = out
            if illegal:
                return out
        else:
            s_prime = out
        legal = self.valid_transitions[s, s_prime]
        if legal:
            return out
        return s, True
        # x, y = self.state_to_coor(s)
        # x_prime, y_prime = self.state_to_coor(s_prime)
        # for obstacle in self.obstacles_list:
        #     illegal = is_valid(x, y, x_prime, y_prime, obstacle)
        #     if illegal:
        #         return s, True
        # return out

    def q_values(self, goal_spec, forwards_value_iter=_value_iter,
            goal_stuck=False):
        """
        Calculate the hardmax Q values for each state action pair.
        For GridWorldMDPs, the goal_spec is simply the goal state.

        Params:
            goal_spec [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If this is True, then all actions other than
                ABSORB are illegal in the goal state.

        Returns:
            Q [np.ndarray]: An SxA array containing the q values
                corresponding to each (s, a) pair.
        """
        if (goal_spec, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_spec, goal_stuck)])

        V = forwards_value_iter(self, goal_spec)

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in self.valid_states: # range(self.S):
            if s == goal_spec and goal_stuck:
                Q[s, Directions.ABSORB] = 0
                # All other actions will be -np.inf by default.
                continue

            for a in range(self.A):
                if s == goal_spec and a == Directions.ABSORB:
                    Q[s, a] = 0
                else:
                    Q[s, a] = self.rewards[s, a] - self.action_cost(a) + V[self.transition(s, a)]
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_spec, goal_stuck)] = Q
        return np.copy(Q)

    def action_cost(self, a):
        step_size, direction = self.action_converter[a]
        return self.step_cost * step_size

def is_valid(x, y, x_prime, y_prime, obstacle):
    (x1, y1), (x2, y2) = obstacle
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Trivial cases: they both on the same side of the obstacle
    if x < x_min and x_prime < x_min:
        return True
    if x > x_max and x_prime > x_max:
        return True
    if y < y_min and y_prime < y_min:
        return True
    if y > y_max and y_prime > y_max:
        return True

    # Less trivial, use the line to split the space in two, and make sure that all four corners are on the same side
    label = None # Area
    for point in [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]:
        u = np.array([point[0] - x, point[1] - y])
        v = np.array([x_prime - x, y_prime - y])
        if label is None:
            label = np.cross(u, v)
        else:
            if np.cross(u, v) * label < 0:
                return False
    return True



