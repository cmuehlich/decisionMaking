import numpy as np
import copy
from typing import List, Dict
from agents.baseClass import Agent
from env.stateSpace import State, STATE_SPACE_TYPE
from env.dynamics import TransitionModel
from env.reward import ExplicitReward
from enum import Enum

class MDP_SOLVER(Enum):
    VALUE_ITERATION = 0
    POLICY_ITERATION = 1

class MDPAgent(Agent):
    def __init__(self, config: Dict):
        super().__init__(config=config,
                         state_space_type=STATE_SPACE_TYPE.DISCRETE,
                         reward_model=ExplicitReward(min_reward=-1, max_reward=1),
                         system_dynamics=TransitionModel(),
                         learning_type=None,
                         initial_target_policy=None,
                         initial_behavior_policy=None)

        self.convergence = config["convergenc_criteria"]

        if config["solver"] == "value_iteration":
            self.solver = MDP_SOLVER.VALUE_ITERATION
        elif config["solver"] == "policy_iteration":
            self.solver = MDP_SOLVER.POLICY_ITERATION
        else:
            raise IOError("Choose available solver!")

        # Value Space Definitions
        self.value_space = np.zeros(shape=(self.state_dim, self.state_dim))

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

    def policy_iteration(self) -> np.ndarray:
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
            if check_sum/(self.state_dim**2) >= 0.99:
                policy_is_stable = True
                print("Policy is stable!")

        # Save policy
        self.target_policy = policy

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

    def policy_improvement(self) -> np.ndarray:
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
        self.target_policy = optimal_policy

        return optimal_policy

    def choose_action(self, state: State) -> int:
        """
        Function to retrieve the best action for the encountered state. The state needs to be first transformed from
        the continous to the discrete space.
        :param policy: Array of shape=(State_dim, State_dim)
        :param state: Tuple of (Position, Acceleration)
        :return: Action Space Idx, 0: Deaccelerate, 1: Hold, 2: Accelerate.
        """
        return int(self.target_policy[state.x_idx][state.v_idx])

    def get_future_reward(self, state: State, action: int) -> float:
        """
        Computation of future reward which will be obtained by taking action a in state s. For determinisitic system
        dynamics.
        :param state:
        :param action:
        :return: Reward of action a taken in state s
        """
        s_t1_obs = self.transition_model.state_transition(state, action)
        x_t1_idx, v_t1_idx = self.env.get_state_space_idx(observation=s_t1_obs)
        new_state = State(x=s_t1_obs[0], v=s_t1_obs[1], x_pos=x_t1_idx, v_pos=v_t1_idx)
        reward = self.value_space[new_state.x_idx][new_state.v_idx]
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
