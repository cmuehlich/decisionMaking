import numpy as np
import math
import copy
import itertools

class MDPAgent():
    def __init__(self, config):
        # Parameter Definitions
        self.state_dim = config["state_dim"]
        self.convergence = config["convergenc_criteria"]
        self.discount_factor = config["discount_factor"]

        # Model Definitions
        self.env = StateSpace(self.state_dim)
        self.reward_model = RewardModel()
        self.transition_model = TransitionModel()

        # Space Definitions
        self.state_space_idx = self.env.get_state_space()
        self.value_space = np.zeros(shape=(self.state_dim, self.state_dim))
        self.action_space = [0, 1, 2]

    def compute_deterministic_cum_reward(self, state):
        """
        Computes the cumulative reward for deterministic system dynamics.
        :param state: Tuple of (Position, Acceleration)
        :return: Array of cumulative rewards for each action of action space
        """
        return [self.reward_model.get_reward(state, action) + self.discount_factor *
                self.get_future_reward(state, action) for action in self.action_space]

    def compute_expected_cum_reward(self, state, probabilities):
        """
        Computes the cumulative reward for stochastic system dynamics.
        """
        pass


    def value_iteration(self):
        """
        Value Iteration procedure. System dynamics are deterministic why probabilities can be omitted here. If system
        involves a stochastic process, for each action the expectation of cumulative rewards needs to be computed by
        weighting each possible outcome by their state transition probability.
        :return:
        """
        print("Starting Value Iteration...")
        counter = 0
        valid = True
        while valid:
            max_diff = 0
            counter +=1
            value_space_copy = copy.deepcopy(self.value_space)
            for i, j in self.state_space_idx:
                state = self.env.get_state_space_value(i, j)
                max_v = np.max(self.compute_deterministic_cum_reward(state))
                # UPDATE STATE
                self.value_space[i][j] = max_v
                max_diff = max(max_diff, np.abs(max_v - value_space_copy[i][j]))

            # CHECK CONVERGENCE
            if max_diff < self.convergence:
                valid = False
            print("Running Iteration {}:".format(counter))
            print("ERROR: {}".format(max_diff))
        print("Value Iteration converged!")

    def policy_iteration(self):
        """
        Policy Iteration procedure. System dynamics are deterministic why probabilities can be omitted here. If system
        involves a stochastic process, for each action the expectation of cumulative rewards needs to be computed by
        weighting each possible outcome by their state transition probability.
        :return:
        """
        policy_is_stable = False

        # Init policy and value function
        policy = np.zeros_like(self.value_space)
        for i, j in self.state_space_idx:
            policy[i][j] = np.random.choice(self.action_space)

        # Iterate until policy is stable
        while not policy_is_stable:
            check_sum = 0
            self.policy_evaluation(policy)
            for i, j in self.state_space_idx:
                old_action = policy[i][j]
                state = self.env.get_state_space_value(i, j)
                new_action = np.argmax(self.compute_deterministic_cum_reward(state))
                policy[i][j] = new_action
                if old_action == new_action:
                    check_sum += 1
            print("Policy changing rate: {}".format(check_sum/(self.state_dim**2)))
            if check_sum == (self.state_dim ** 2):
                policy_is_stable = True
                print("Policy is stable!")
        return policy

    def policy_evaluation(self, policy):
        """
        Policy evaluation. Takes a policy as input and evaluates it by picking actions according to it and updating
        the state value. Policy is deterministic why no expectation needs to be computed
        :param policy: Numpy Array of shape=(State_dim, State_dim)
        :return:
        """
        counter = 0
        valid = True
        while valid:
            max_diff = 0
            counter += 1
            value_space_copy = copy.deepcopy(self.value_space)
            for i, j in self.state_space_idx:
                state = self.env.get_state_space_value(i, j)
                action = policy[i][j]
                vs = self.reward_model.get_reward(state, action) + self.discount_factor*self.get_future_reward(state, action)
                # UPDATE STATE
                self.value_space[i][j] = vs
                max_diff = max(max_diff, np.abs(vs - value_space_copy[i][j]))

            # CHECK CONVERGENCE
            if max_diff < self.convergence:
                valid = False
            print("Running Iteration {}:".format(counter))
            print("ERROR: {}".format(max_diff))
        print("Value Function Update converged!")

    def policy_improvement(self):
        """
        Policy improvement updates the policy greedily w.r.t. cumulative reward. This function is used for making the
        final improvement after value iteration. Policy iteration will use its own integrated form of policy improvement
        because of computational efficiency.
        :return: Numpy Array of shape=(State_dim, State_dim)
        """
        optimal_policy = np.zeros_like(self.value_space)
        for i, j in self.state_space_idx:
            state = self.env.get_state_space_value(i, j)
            best_action = np.argmax(self.compute_deterministic_cum_reward(state))
            # UPDATE POLICY
            optimal_policy[i][j] = best_action

        return optimal_policy

    def get_optimal_action(self, policy, state):
        """
        Function to retrieve the best action for the encountered state. The state needs to be first transformed from
        the continous to the discrete space.
        :param policy: Array of shape=(State_dim, State_dim)
        :param state: Tuple of (Position, Acceleration)
        :return: Action Space Idx, 0: Deaccelerate, 1: Hold, 2: Accelerate.
        """
        x_idx, v_idx = self.env.get_state_space_idx(state)
        return int(policy[x_idx][v_idx])

    def get_future_reward(self, s, a):
        """
        Computation of future reward which will be obtained by taking action a in state s. For determinisitic system
        dynamics.
        :param s:
        :param a:
        :return:
        """
        reward = 0.0
        s_t1 = self.transition_model.state_transition(s, a)
        if self.env.check_bounds(s_t1):
            x_idx, v_idx = self.env.get_state_space_idx(s_t1)
            reward = self.value_space[x_idx][v_idx]
        return reward

    def rms(self, vs, vs_copy):
        """
        RMS-Error for convergence check.
        :param vs:
        :param vs_copy:
        :return:
        """
        vs_sqdiff = np.power(np.subtract(vs, vs_copy), 2)
        vs_mean = np.sum(vs_sqdiff) / self.state_dim**2
        rms = np.sqrt(vs_mean)
        return rms

class StateSpace():
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

class TransitionModel():
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
