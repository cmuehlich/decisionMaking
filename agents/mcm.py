from typing import List, Dict
import numpy as np
import copy
from agents.baseClass import Agent, Experience, POLICY_TYPES, POLICY_LEARNING_TYPES
from env.stateSpace import State, STATE_SPACE_TYPE


class MCMAgent(Agent):
    def __init__(self, config: Dict,
                 state_space_type: STATE_SPACE_TYPE = STATE_SPACE_TYPE.DISCRETE,
                 policy_learning_type: POLICY_LEARNING_TYPES = POLICY_LEARNING_TYPES.OFFLINE,
                 initial_target_policy: POLICY_TYPES = POLICY_TYPES.EPS_GREEDY,
                 initial_behavior_policy: POLICY_TYPES = POLICY_TYPES.EPS_GREEDY):
        super().__init__(config=config,
                         state_space_type=state_space_type,
                         reward_model=None,
                         system_dynamics=None,
                         learning_type=policy_learning_type,
                         initial_target_policy=initial_target_policy,
                         initial_behavior_policy=initial_behavior_policy)

        if POLICY_TYPES.FUNC in [initial_behavior_policy, initial_target_policy]:
            self.env.create_tilings()
            # Init q-value function with zero weights
            self.q_space = [np.zeros(self.env.feature_size) for _ in range(len(self.action_space))]
        else:
            # Space of state-action values
            self.q_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
        # Storage for averaging returns
        self.returns = np.zeros_like(self.q_space)
        self.visits = np.zeros_like(self.returns)

        # MCM specific Definitions
        if config["mc_variant"] not in ["first_visit", "all_visit"]:
            raise IOError("Choose one of the available MC variants")
        self.mcm_type = config["mc_variant"]
        self.experience: List[Experience] = list()
        self.history = dict()

    def check_first_visit(self, check_state: State, reverse_index: int) -> bool:
        first_visited = True

        trajectory = copy.deepcopy(self.experience)
        for _ in range(reverse_index+1):
            trajectory.pop(-1)
        for observation in reversed(trajectory):
            if observation.state.x_idx == check_state.x_idx and observation.state.y_idx == check_state.y_idx:
                first_visited = False
                break

        return first_visited

    def mc_control(self, episode: int = 0) -> None:
        """
        Main function for Monte Carlo Control.
        :param episode: number of episode
        :return:
        """
        if self.mcm_type == "first_visit":
            cum_reward = self._mc_control_first_visit()
        else:
            cum_reward = self._mc_control_all_visits()

        # Save episode results
        self.history[episode] = cum_reward
        # Refresh experience
        self.experience.clear()

    def _mc_control_all_visits(self) -> float:
        """
        Monte Carlo Control algorithm
        :param episode: number of episode - all visits
        :return: cumulative reward
        """
        cum_reward = 0

        # Loop through experience in reverse
        for observation in reversed(self.experience):
            cum_reward = observation.reward + self.discount_factor * cum_reward

            # MC all visits
            # Update Q-Value, if behavior and target policy are not equal probability distribution
            # importance sampling is required
            self.update_func(observation=observation, cumulative_reward=cum_reward)

        return cum_reward

    def _mc_control_first_visit(self) -> float:
        """
        Monte Carlo Control algorithm - first visit
        :return: cumulative reward
        """
        cum_reward = 0
        counter = 0

        # Loop through experience in reverse
        for observation in reversed(self.experience):
            cum_reward = observation.reward + self.discount_factor * cum_reward

            # MC first visits
            if not self.check_first_visit(check_state=observation.state, reverse_index=counter):
                counter += 1
                continue
            else:
                # Update Q-Value, if behavior and target policy are not equal probability distribution
                # importance sampling is required
                self.update_func(observation=observation, cumulative_reward=cum_reward)

            # Increment counter
            counter += 1

        return cum_reward

    def update_func(self, observation: Experience, cumulative_reward: float) -> None:
        self.state_update(observation=observation, cumulative_reward=cumulative_reward)
        self._average_return(observation=observation)

    def _average_return(self, observation: Experience) -> None:
        self.q_space[observation.action.value, observation.state.x_idx, observation.state.y_idx] = \
            float(self.returns[observation.action.value, observation.state.x_idx, observation.state.y_idx]
                  / self.visits[observation.action.value, observation.state.x_idx, observation.state.y_idx])

    def state_update(self, observation: Experience, cumulative_reward: float) -> None:
        self.visits[observation.action.value, observation.state.x_idx, observation.state.y_idx] += 1
        self.returns[observation.action.value, observation.state.x_idx, observation.state.y_idx] += cumulative_reward

    def choose_action(self, state: State) -> int:
        # Constant epsilon greedy policy for behavior generation
        if self.behavior_policy.policy_type == POLICY_TYPES.FUNC:
            feature_vec = self.env.get_feature_vector(observation=[state.x, state.y])
        else:
            feature_vec = None
        return self.behavior_policy.get_action(state=state, epsilon=self.epsilon,
                                               q_space=self.q_space, feature_vector=feature_vec)
