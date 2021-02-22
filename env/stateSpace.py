from typing import Tuple, List, Dict, Union
import itertools
from enum import Enum
import numpy as np

class STATE_SPACE_TYPE(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class ACTION_SPACE_TYPE(Enum):
    DISCRETE = [0, 1, 2]
    CONTINUOUS = np.inf


class ACTION_SPACE(Enum):
    DECELERATE = 0
    IDLE = 1
    ACCELERATE = 2


class Action:
    def __init__(self, a: int):
        self.a = a


class State:
    def __init__(self, x: float, v: float, x_pos: Union[int, None] = None, v_pos: Union[int, None] = None):
        self.x = x
        self.v = v

        self.x_idx = x_pos
        self.v_idx = v_pos

    def get_state_value(self) -> Tuple[float, float]:
        return self.x, self.v

    def get_state_idx(self) -> Tuple[int, int]:
        return self.x_idx, self.v_idx


class StateSpace:
    def __init__(self):
        self.state_params = {"v_min": -0.07,
                             "v_max": 0.07,
                             "x_min": -1.2,
                             "x_max": 0.6}

    def check_bounds(self, state: State) -> bool:
        """
        Checks whether state is within the discretized state space.
        :param state: State to be checked
        :return: Boolean specifiying validity
        """
        valid = False
        if self.state_params["x_min"] <= state.x <= self.state_params["x_max"] and \
                self.state_params["v_min"] <= state.v <= self.state_params["v_max"]:
            valid = True
        return valid

    def get_terminal_states(self) -> List[Tuple[int, int]]:
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

    @staticmethod
    def is_terminal(state: State) -> bool:
        if state.x >= 0.5:
            return True
        else:
            return False


class DiscreteStateSpace(StateSpace):
    """
    Class for handling everything state space related. In particular the consistent mapping between continous and
    discrete state space
    """
    def __init__(self, n: int):
        super().__init__()
        self.state_dim = n

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
        :param observation:
        :return: Tuple of state indexes
        """
        x_n = (observation[0]-self.state_params["x_min"])/self.x_size
        x_idx = int(x_n - (x_n % self.x_size))
        v_n = (observation[1] - self.state_params["v_min"]) / self.v_size
        v_idx = int(v_n - (v_n % self.v_size))

        assert x_idx <= self.state_dim
        assert v_idx <= self.state_dim
        return x_idx, v_idx


class ContinuousStateSpace(StateSpace):
    def __init__(self):
        super().__init__()
        self.feature_size: int = 0
        self.feature_map: Dict = dict()
        self.tiling_info = dict()

    def create_tilings(self):
        disc_steps = [10, 20]
        x_offset, v_offset = [0.1, 0.01], [0.01, 0.001]

        tiling_counter = 0
        for x_off, v_off in itertools.product(x_offset, v_offset):
            for n_x, n_v in itertools.product(disc_steps, disc_steps):
                x_size = round((self.state_params["x_max"] - self.state_params["x_min"] - x_off) / n_x, 6)
                v_size = round((self.state_params["v_max"] - self.state_params["v_min"] - v_off) / n_v, 6)

                x_min = self.state_params["x_min"] - x_off
                x_steps = [round(x_min + i*x_size, 6) for i in range(n_x)]
                v_min = self.state_params["v_min"] - v_off
                v_steps = [round(v_min + i*v_size, 6) for i in range(n_v)]

                self.feature_size += (len(x_steps) * len(v_steps))
                self.feature_map[tiling_counter] = [(i, j) for i, j in itertools.product(np.arange(n_x), np.arange(n_v))]
                self.tiling_info[tiling_counter] = {"x_lower_bound": x_min,
                                                    "x_cell_size": x_size,
                                                    "x_steps": n_x,
                                                    "v_lower_bound": v_min,
                                                    "v_cell_size": v_size,
                                                    "v_steps": n_v}
                tiling_counter += 1

    def get_feature_vector(self, observation: List[float]):
        feature_vector = []
        for tiling_id, tiling_dict in self.tiling_info.items():
            tile_feat = self.get_active_feature(observation=observation, tiling_info=tiling_dict)
            tile_feat_idx = self.feature_map[tiling_id].index(tile_feat)
            tile_feature_vector = np.zeros(len(self.feature_map[tiling_id]))
            tile_feature_vector[tile_feat_idx] = 1
            feature_vector = np.concatenate((feature_vector, tile_feature_vector), axis=0)

        return feature_vector

    @staticmethod
    def get_active_feature(observation: List[float], tiling_info: Dict) -> Tuple[int, int]:
        """
        Transforms the state from continous to discrete space by referring to the indexed state values.
        :param observation:
        :param tiling_info:
        :return: Tuple of state indexes
        """
        x_n = (observation[0]-tiling_info["x_lower_bound"]/tiling_info["x_cell_size"])
        x_idx = int(x_n - (x_n % tiling_info["x_cell_size"]))
        v_n = (observation[1] - tiling_info["v_lower_bound"] / tiling_info["v_cell_size"])
        v_idx = int(v_n - (v_n % tiling_info["v_cell_size"]))

        assert x_idx <= tiling_info["x_steps"]
        assert v_idx <= tiling_info["v_steps"]

        return x_idx, v_idx
