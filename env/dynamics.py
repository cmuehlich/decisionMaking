import math
from typing import List
import numpy as np
from env.stateSpace import State

class TransitionModel():
    """
    Class for dealing with system dynamics.
    """
    def __init__(self, _type: str = "deterministic"):
        self.system_params = {"force": 0.001,
                              "gravity": 0.0025,
                              "v_min": -0.07,
                              "v_max": 0.07,
                              "x_min": -1.2,
                              "x_max": 0.6}
        if _type == "deterministic":
            self.state_transition = self._deterministic_state_transition
        elif _type == "stochastic":
            raise NotImplementedError
        else:
            raise IOError("Specified type of transition model does not exist!")

    def _deterministic_state_transition(self, s_t: State, a_t: int) -> List[float]:
        """
        Describes the deterministic state transition/ system dynamics.
        :param s_t: Initial State
        :param a_t: Chosen Action
        :return: Future State
        """
        v_t1 = s_t.y + (a_t - 1) * self.system_params["force"] + math.cos(3 * s_t.x) * (-self.system_params["gravity"])
        v_t1 = np.clip(v_t1, self.system_params["v_min"], self.system_params["v_max"])
        x_t1 = s_t.x + v_t1
        x_t1 = np.clip(x_t1, self.system_params["x_min"], self.system_params["x_max"])

        if (x_t1 == self.system_params["x_min"] and v_t1 < 0):
            v_t1 = 0

        return [x_t1, v_t1]




