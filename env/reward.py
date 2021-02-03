import numpy as np
import abc
from env.stateSpace import State

class RewardModel(abc.ABC):
    """
    Class describing the desired agent behavior.
    """
    def __init__(self, min_reward: float, max_reward: float):
        self.min_reward = min_reward
        self.max_reward = max_reward

    @abc.abstractmethod
    def _get_reward(self, s: State, a: int):
        """
        Placeholder for reward function
        :param s:
        :param a:
        :return:
        """
        raise NotImplementedError

    def get_reward(self, s: State, a: int) -> float:
        """
        Reward function R(s,a) that maps state-action pairs to some scalar value.
        :return:
        """
        return self._get_reward(s, a)

class ExplicitReward(RewardModel):
    def __init__(self, min_reward: float, max_reward: float):
        super().__init__(min_reward, max_reward)

    def _get_reward(self, s: State, a: int) -> float:
        """
        Explicitly defined reward function.
        :param s:
        :param a:
        :return: float containing the obtained reward
        """
        if s.x >= 0.5:
            reward = self.max_reward
        elif s.x < -1.0:
            reward = self.min_reward
        elif np.abs(s.v) >= 0.035 and s.x < -0.3:
            reward = self.min_reward
        else:
            if s.v > 0 and a == 2:
                reward = self.max_reward / 2
            elif s.v > 0 and a != 2:
                reward = self.min_reward
            elif s.v < 0 and a == 0 and s.x < -0.3:
                reward = self.max_reward / 2
            elif s.v < 0 and a != 0 and s.x >= -0.3:
                reward = self.min_reward
            else:
                reward = 0
        return reward