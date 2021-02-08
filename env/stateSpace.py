from typing import Tuple, List
import itertools
from enum import Enum

class ACTION_SPACE_TYPE(Enum):
    DISCRETE = 0
    CONTINOUS = 1

class ACTION_SPACE(Enum):
    DEACCELERATE = 0
    IDLE = 1
    ACCELERATE = 2

class Action():
    def __init__(self, a: int):
        self.a = a

class State():
    def __init__(self, x: float, v: float, x_pos: int, v_pos: int):
        self.x = x
        self.v = v

        self.x_idx = x_pos
        self.v_idx = v_pos

    def get_state(self) -> Tuple[float, float]:
        return self.x, self.v

class StateSpace():
    """
    Class for handling everything state space related. In particular the consistent mapping between continous and
    discrete state space
    """
    def __init__(self, n: int):
        self.state_dim = n
        self.state_params = {"v_min": -0.07,
                             "v_max": 0.07,
                             "x_min": -1.2,
                             "x_max": 0.6}
        self.x_size = round((self.state_params["x_max"] - self.state_params["x_min"]) / n, 6)
        self.v_size = round((self.state_params["v_max"] - self.state_params["v_min"]) / n, 6)

        self.x_steps = [round(self.state_params["x_min"] + i*self.x_size, 6) for i in range(n)]
        self.v_steps = [round(self.state_params["v_min"] + i*self.v_size, 6) for i in range(n)]

    def get_state_space(self) -> Tuple[List[Tuple[int, int]], List[State]]:
        """
        Returns the discretized state space for comfortable iterations.
        :return: List of state space index integers and list of all states
        """
        state_idxs = [int(i) for i in range(self.state_dim)]
        state_space_idx = []
        state_list = []
        for x_idx, v_idx in itertools.product(state_idxs, state_idxs):
            state_space_idx.append((x_idx, v_idx))
            state_list.append(State(x=self.x_steps[x_idx], v=self.v_steps[v_idx],
                                    x_pos=x_idx, v_pos=v_idx))
        return state_space_idx, state_list

    def get_state_space_value(self, x_idx: int, v_idx: int) -> State:
        """
        Transforms the state from discrete to continous space by choosing the centers of each cell.
        :param x_idx: Int index of discretized Position
        :param v_idx: Int index of discretized Position
        :return: State in continous space
        """
        # Get center value of each cell
        x = self.x_steps[x_idx] + 0.5 * self.x_size
        v = self.v_steps[v_idx] + 0.5 * self.v_size

        return State(x=x, v=v, x_pos=x_idx, v_pos=v_idx)


    def get_state_space_idx(self, observation: List[float]) -> Tuple[int, int]:
        """
        Transforms the state from continous to discrete space by referring to the indexed state values.
        :param state:
        :return: Tuple of state indexes
        """
        x_n = (observation[0]-self.state_params["x_min"])/self.x_size
        x_idx = int(x_n - (x_n % self.x_size))
        v_n = (observation[1] - self.state_params["v_min"]) / self.v_size
        v_idx = int(v_n - (v_n % self.v_size))

        assert x_idx <= self.state_dim
        assert v_idx <= self.state_dim
        return x_idx, v_idx

    def check_bounds(self, state: State) -> bool:
        """
        Checks whether state is within the discretized state space.
        :param state: State to be checked
        :return: Boolean specifiying validity
        """
        valid = False
        if state.x <= self.state_params["x_max"] and state.x >= self.state_params["x_min"] and \
                state.v <= self.state_params["v_max"] and state.v >= self.state_params["v_min"]:
            valid = True
        return valid

    def get_terminal_states(self) -> List[Tuple[int]]:
        """
        Returns a list of state indexes which belong to the terminal states of the mountain car setting
        :return: List of state space indexes
        """
        v_state_idxs = [int(i) for i in range(self.state_dim)]
        goal_states = []
        for idx, x in enumerate(self.x_steps):
            if x >= 0.5:
                for i, j in itertools.product([idx], v_state_idxs):
                    goal_states.append((i, j))
        return goal_states