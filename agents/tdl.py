from typing import Dict, Tuple
from enum import Enum
from agents.baseClass import Agent
from env.stateSpace import State, Action
import numpy as np
from agents.baseClass import POLICY_TYPES, POLICY_LEARNING_TYPES

class TDL_METHODS(Enum):
    SARSA = 0
    EXPECTED_SARSA = 1
    Q_LEARNING = 2
    DOUBLE_Q_LEARNING = 3

class OPERATORS(Enum):
    MAXIMIZATION = 0
    AVERAGE = 1
    SUM = 2

class TDLAgent(Agent):
    def __init__(self, config: Dict, tdl_method: TDL_METHODS):
        self.tdl_type = tdl_method
        self.update_rule_info = {
            TDL_METHODS.SARSA: {"learning_type": POLICY_LEARNING_TYPES.ONLINE,
                                "target_policy": POLICY_TYPES.EPS_GREEDY,
                                "behavior_policy": None
                                },
            TDL_METHODS.EXPECTED_SARSA: {"learning_type": POLICY_LEARNING_TYPES.OFFLINE,
                                         "target_policy": POLICY_TYPES.EPS_GREEDY,
                                         "behavior_policy": POLICY_TYPES.EPS_GREEDY
                                         },
            TDL_METHODS.Q_LEARNING: {"learning_type": POLICY_LEARNING_TYPES.OFFLINE,
                                     "target_policy": POLICY_TYPES.EPS_GREEDY,
                                     "behavior_policy": POLICY_TYPES.EPS_GREEDY
                                    },
            TDL_METHODS.DOUBLE_Q_LEARNING: {"learning_type": POLICY_LEARNING_TYPES.OFFLINE,
                                            "target_policy": POLICY_TYPES.EPS_GREEDY,
                                            "behavior_policy": POLICY_TYPES.EPS_GREEDY
                                            }
        }
        self.update_rule_dict = {TDL_METHODS.SARSA: self.sarsa_update,
                                 TDL_METHODS.EXPECTED_SARSA: self.expected_sarsa_update,
                                 TDL_METHODS.Q_LEARNING: self.q_learning_update,
                                 TDL_METHODS.DOUBLE_Q_LEARNING: self.double_q_learning_update}

        super().__init__(config=config,
                         reward_model=None,
                         system_dynamics=None,
                         learning_type=self.update_rule_info[tdl_method]["learning_type"],
                         initial_target_policy=self.update_rule_info[tdl_method]["target_policy"],
                         initial_behavior_policy=self.update_rule_info[tdl_method]["behavior_policy"])

        # TD specific parameters
        self.learning_rate = config["td_alpha"]
        self.update_func = self.update_rule_dict[tdl_method]
        if config["dq_learning_merging_operator"] == "sum":
            self.double_q_merging_operator = OPERATORS.SUM
        elif config["dq_learning_merging_operator"] == "average":
            self.double_q_merging_operator = OPERATORS.AVERAGE
        elif config["dq_learning_merging_operator"] == "max":
            self.double_q_merging_operator = OPERATORS.MAXIMIZATION
        else:
            raise IOError("Choose one of the available merging operations")

        # Space of state-action values - Init randomly
        q_spaces = []
        if self.tdl_type == TDL_METHODS.DOUBLE_Q_LEARNING:
            self.q1_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
            q_spaces.append(self.q1_space)
            self.q2_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
            q_spaces.append(self.q2_space)
        else:
            self.q_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
            q_spaces.append(self.q_space)

        # Set terminal states to zero
        for x_idx, v_idx in self.env.get_terminal_states():
            for action_id in np.arange(len(self.action_space)):
                for q_space in q_spaces:
                    q_space[action_id][x_idx][v_idx] = 0

    def combine_q_spaces(self, q1_space: np.ndarray, q2_space: np.ndarray, method: OPERATORS = OPERATORS.SUM) -> np.ndarray:
        q_space = np.zeros_like(q1_space)
        if method == OPERATORS.SUM:
            # Element-wise sum of q_spaces
            for action_id in self.action_space:
                q_space[action_id, :, :] = np.add(q1_space[action_id, :, :], q2_space[action_id, :, :])
        elif method == OPERATORS.AVERAGE:
            # Average of q_spaces
            for action_id in self.action_space:
                for i, j in self.state_space_idx:
                    q_space[action_id, i, j] = np.mean([q1_space[action_id, i, j], q2_space[action_id, i, j]])
        elif method == OPERATORS.MAXIMIZATION:
            # Max value of q_spaces
            for action_id in self.action_space:
                for i, j in self.state_space_idx:
                    q_space[action_id, i, j] = np.max([q1_space[action_id, i, j], q2_space[action_id, i, j]])

        return q_space

    def choose_action(self, state: State) -> int:
        # if double q-learning the policy has two action value spaces to consider which we can mix in advance
        if self.tdl_type == TDL_METHODS.DOUBLE_Q_LEARNING:
            q_space = self.combine_q_spaces(q1_space=self.q1_space, q2_space=self.q2_space,
                                            method=self.double_q_merging_operator)
        else:
            q_space = self.q_space

        # if on-policy use same policy for behavior generation as for update
        if self.learning_type == POLICY_LEARNING_TYPES.ONLINE:
            # Constant epsilon
            return self.target_policy.eps_greedy_policy(state=state, epsilon=self.epsilon, q_space=q_space)
        else:
            return self.behavior_policy.eps_greedy_policy(state=state, epsilon=self.epsilon, q_space=q_space)

    def sarsa_update(self, state_t0: State, action_t0: Action, reward: float, state_t1: State, action_t1: Action) -> None:
        q_v_prev = self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx]
        td_error = reward + self.discount_factor * self.q_space[action_t1.a][state_t1.x_idx][state_t1.v_idx] - q_v_prev

        self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx] += self.learning_rate * td_error

    def expected_sarsa_update(self, state_t0: State, action_t0: Action, reward: float, state_t1: State) -> None:
        q_v_prev = self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx]
        q_values = [self.q_space[action_id][state_t1.x_idx][state_t1.v_idx] for action_id in self.action_space]
        max_q_action = np.argmax(q_values)

        # For constant epsilon, compute expected future reward
        max_prob, min_prob = self.behavior_policy.get_eps_greedy_probabilities(epsilon=self.epsilon)
        expected_future_reward = 0
        for action_id in self.action_space:
            if action_id == max_q_action:
                expected_future_reward += max_prob * q_values[action_id]
            else:
                expected_future_reward += min_prob * q_values[action_id]

        td_error = reward + self.discount_factor * expected_future_reward - q_v_prev

        self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx] += self.learning_rate * td_error

    def q_learning_update(self, state_t0: State, action_t0: Action, reward: float, state_t1: State) -> None:
        q_v_prev = self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx]

        max_q_value_t1 = np.max([self.q_space[action_id][state_t1.x_idx][state_t1.v_idx]
                                 for action_id in self.action_space])
        td_error = reward + self.discount_factor * max_q_value_t1 - q_v_prev

        self.q_space[action_t0.a][state_t0.x_idx][state_t0.v_idx] += self.learning_rate * td_error

    def double_q_learning_update(self, state_t0: State, action_t0: Action, reward: float, state_t1: State) -> None:
        # With probability of 0.5 update q1 space, otherwise q2
        if np.random.uniform() > 0.5:
            q1_v_prev = self.q1_space[action_t0.a][state_t0.x_idx][state_t0.v_idx]
            max_q1_action = np.argmax([self.q1_space[action_id][state_t1.x_idx][state_t1.v_idx]
                                       for action_id in self.action_space])
            q2_action_value_t1 = self.q2_space[max_q1_action][state_t1.x_idx][state_t1.v_idx]
            td_error = reward + self.discount_factor * q2_action_value_t1 - q1_v_prev

            self.q1_space[action_t0.a][state_t0.x_idx][state_t0.v_idx] += self.learning_rate * td_error
        else:
            q2_v_prev = self.q2_space[action_t0.a][state_t0.x_idx][state_t0.v_idx]
            max_q2_action = np.argmax([self.q2_space[action_id][state_t1.x_idx][state_t1.v_idx]
                                       for action_id in self.action_space])
            q1_action_value_t1 = self.q2_space[max_q2_action][state_t1.x_idx][state_t1.v_idx]
            td_error = reward + self.discount_factor * q1_action_value_t1 - q2_v_prev

            self.q2_space[action_t0.a][state_t0.x_idx][state_t0.v_idx] += self.learning_rate * td_error


