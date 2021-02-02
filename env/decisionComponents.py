import math

class TransitionModel():
    """
    Class for dealing with system dynamics.
    """
    def __init__(self):
        self.force = 0.001
        self.gravity = 0.0025

    def state_transition(self, s_t, a_t):
        """
        Describes the deterministic state transition/ system dynamics.
        :param s_t: Initial State
        :param a_t: Chosen Action
        :return: Future State
        """
        x_pos, vel = s_t
        v_t1 = vel + (a_t - 1) * self.force + math.cos(3 * x_pos) * (-self.gravity)
        v_t1 = np.clip(v_t1, -0.07, 0.07)
        x_t1 = x_pos + v_t1
        x_t1 = np.clip(x_t1, -1.2, 0.6)

        if (x_t1 == -1.2 and v_t1 < 0):
            v_t1 = 0

        return x_t1, v_t1

class RewardModel():
    """
    Class describing the desired agent behavior.
    """
    def __init__(self):
        self.min_reward = -1
        self.max_reward = 1

    def get_reward(self, s, a):
        """
        Reward function R(s,a) that maps state-action pairs to some scalar value.
        :return:
        """
        x, v = s

        if x >= 0.5:
            reward = self.max_reward
        elif x < -1.0:
            reward = self.min_reward
        elif np.abs(v) >= 0.035 and x < -0.3:
            reward = self.min_reward
        else:
            # check direction of acceleration
            if v > 0 and a == 2:
                reward = self.max_reward / 2
            elif v > 0 and a != 2:
                reward = self.min_reward
            elif v < 0 and a == 0 and x < -0.3:
                reward = self.max_reward / 2
            elif v < 0 and a != 0 and x >= -0.3:
                reward = self.min_reward
            else:
                reward = 0
        return reward
