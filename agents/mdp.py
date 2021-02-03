import numpy as np
import copy
from typing import List, Dict
from env.stateSpace import StateSpace, State
from env.dynamics import TransitionModel
from env.reward import ExplicitReward

class MDPAgent():
    def __init__(self, config: Dict):
        # Parameter Definitions
        self.state_dim = config["state_dim"]
        self.convergence = config["convergenc_criteria"]
        self.discount_factor = config["discount_factor"]

        # Model Definitions
        self.env = StateSpace(self.state_dim)
        self.reward_model = ExplicitReward(min_reward=-1, max_reward=1)
        self.transition_model = TransitionModel()

        # Space Definitions
        self.state_space_idx = self.env.get_state_space()
        self.value_space = np.zeros(shape=(self.state_dim, self.state_dim))
        self.action_space = [0, 1, 2]
        self.policy = None

    def compute_deterministic_cum_reward(self, state: State) -> List[float]:
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


    def value_iteration(self) -> None:
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

    def policy_iteration(self) -> List[List[float]]:
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

        # Save policy
        self.policy = policy

        return policy

    def policy_evaluation(self, policy: List[List[float]]) -> None:
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

    def policy_improvement(self) -> List[List[int]]:
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
            optimal_policy[i][j] = int(best_action)

        # Save policy
        self.policy = optimal_policy

        return optimal_policy

    def choose_action(self, state: State) -> int:
        """
        Function to retrieve the best action for the encountered state. The state needs to be first transformed from
        the continous to the discrete space.
        :param policy: Array of shape=(State_dim, State_dim)
        :param state: Tuple of (Position, Acceleration)
        :return: Action Space Idx, 0: Deaccelerate, 1: Hold, 2: Accelerate.
        """
        x_idx, v_idx = self.env.get_state_space_idx(state)
        return int(self.policy[x_idx][v_idx])

    def get_future_reward(self, state: State, action: int) -> float:
        """
        Computation of future reward which will be obtained by taking action a in state s. For determinisitic system
        dynamics.
        :param state:
        :param action:
        :return: Reward of action a taken in state s
        """
        reward = 0.0
        s_t1 = self.transition_model.state_transition(state, action)
        if self.env.check_bounds(s_t1):
            x_idx, v_idx = self.env.get_state_space_idx(s_t1)
            reward = self.value_space[x_idx][v_idx]
        return reward

    def rms(self, vs: List[List[float]], vs_copy: List[List[float]]) -> float:
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
