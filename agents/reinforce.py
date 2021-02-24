from typing import Dict
from env.stateSpace import STATE_SPACE_TYPE
from agents.baseClass import Experience, POLICY_TYPES, POLICY_LEARNING_TYPES
from agents.mcm import MCMAgent

import numpy as np


class Reinforce(MCMAgent):
    def __init__(self, config: Dict,
                 state_space_type: STATE_SPACE_TYPE = STATE_SPACE_TYPE.CONTINUOUS,
                 policy_learning_type: POLICY_LEARNING_TYPES = POLICY_LEARNING_TYPES.OFFLINE,
                 initial_target_policy: POLICY_TYPES = POLICY_TYPES.FUNC,
                 initial_behavior_policy: POLICY_TYPES = POLICY_TYPES.FUNC):
        super().__init__(config=config, state_space_type=state_space_type, policy_learning_type=policy_learning_type,
                         initial_target_policy=initial_target_policy, initial_behavior_policy=initial_behavior_policy)

        # learned state value function
        self.state_value_function = np.zeros(self.env.feature_size)
        self.baseline_learning_rate = config["baseline_learning_rate"]

    # override update function
    def update_func(self, observation: Experience, cumulative_reward: float) -> None:
        self._gradient_update(observation=observation, cumulative_reward=cumulative_reward)

    def _gradient_update(self, observation: Experience, cumulative_reward: float):
        feature_vector = self.env.get_feature_vector(observation=[observation.state.x, observation.state.y])
        action_probs = self.target_policy.action_preference_distribution(q_space=self.q_space,
                                                                         feature_vector=feature_vector)
        baseline = self.state_value_function.dot(feature_vector)
        reward_estimate = cumulative_reward - baseline
        # Avoid division by zero
        if action_probs[observation.action.value] > 0:
            learning_factor = self.learning_rate * self.discount_factor * reward_estimate * (1/action_probs[observation.
                                                                                             action.value])
        else:
            learning_factor = 0

        gradient = feature_vector

        # Update Baseline estimator
        self.state_value_function = np.add(self.state_value_function,
                                           np.multiply(gradient, self.baseline_learning_rate * reward_estimate))
        # Update policy
        self.q_space[observation.action.value] = np.add(self.q_space[observation.action.value],
                                                  np.multiply(gradient, learning_factor))
