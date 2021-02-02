import itertools

class StateSpace():
    """
    Class for handling everything state space related. In particular the consistent mapping between continous and
    discrete state space
    """
    def __init__(self, N):
        self.state_dim = N
        self.x_min = -1.2
        self.x_max = 0.6
        self.v_min = -0.07
        self.v_max = 0.07
        self.x_size = round((self.x_max - self.x_min) / N, 6)
        self.v_size = round((self.v_max - self.v_min) / N, 6)

        self.x_steps = [round(self.x_min + i*self.x_size, 6) for i in range(N)]
        self.v_steps = [round(self.v_min + i*self.v_size, 6) for i in range(N)]

    def get_state_space(self):
        """
        Returns the discretized state space for comfortable iterations.
        :return:
        """
        state_idxs = [i for i in range(self.state_dim)]
        state_space_idx = []
        for x, v in itertools.product(state_idxs, state_idxs):
            state_space_idx.append((x,v))
        return state_space_idx

    def get_state_space_value(self, x_idx, v_idx):
        """
        Transforms the state from discrete to continous space by choosing the centers of each cell.
        :param x_idx: Int index of discretized Position
        :param v_idx: Int index of discretized Position
        :return: State in continous space
        """
        # Get center value of each cell
        x = self.x_steps[x_idx] + 0.5 * self.x_size
        v = self.v_steps[v_idx] + 0.5 * self.v_size
        return x, v


    def get_state_space_idx(self, state):
        """
        Transforms the state from continous to discrete space by referring to the indexed state values.
        :param state:
        :return:
        """
        x, v = state
        x_n = (x-self.x_min)/self.x_size
        x_idx = int(x_n - (x_n % self.x_size))
        v_n = (v - self.v_min) / self.v_size
        v_idx = int(v_n - (v_n % self.v_size))

        assert x_idx <= self.state_dim
        assert v_idx <= self.state_dim
        return x_idx, v_idx

    def check_bounds(self, state):
        """
        Checks whether state is within the discretized state space.
        :param state:
        :return:
        """
        x, v = state
        valid = False
        if x <= self.x_max and x >= self.x_min and v <= self.v_max and v >= self.v_min:
            valid = True
        return valid