from typing import List, Dict
import numpy as np
import copy
from agents.baseClass import Agent, Experience, POLICY_TYPES, POLICY_LEARNING_TYPES
from env.stateSpace import State


class MCMAgent(Agent):
    def __init__(self, config: Dict):
        super().__init__(config=config,
                         reward_model=None,
                         system_dynamics=None,
                         learning_type=POLICY_LEARNING_TYPES.OFFLINE,
                         initial_target_policy=POLICY_TYPES.RANDOM,
                         initial_behavior_policy=POLICY_TYPES.EPS_GREEDY)

        # Space of state-action values
        self.q_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
        # Storage for averaging returns
        self.returns = np.zeros_like(self.q_space)
        self.visits = np.zeros_like(self.returns)

        # MCM specific Definitions
        self.mcm_type = config["solver"]
        self.experience: List[Experience] = list()
        self.history = dict()

        if self.mcm_type == "mcm_first_visit":
            self.mc_control = self._mc_control_first_visit
        elif self.mcm_type == "mcm_all_visit":
            self.mc_control = self._mc_control_all_visits
        else:
            raise IOError("Pick one of the available MCM Variants!")

    def check_first_visit(self, check_state: State, reverse_index: int) -> bool:
        first_visited = True

        trajectory = copy.deepcopy(self.experience)
        trajectory.pop(len(self.experience) - reverse_index - 1)
        for observation in reversed(trajectory):
            if observation.state.x_idx == check_state.x_idx and observation.state.v_idx == check_state.v_idx:
                first_visited = False
                break

        return first_visited

    def _mc_control_all_visits(self, episode: int = 0) -> None:
        """
        Monte Carlo Control algorithm
        :param episode: number of episode - all visits
        :return:
        """
        cum_reward = 0

        # Loop through experience in reverse
        for observation in reversed(self.experience):
            cum_reward = observation.reward + self.discount_factor * cum_reward

            # MC all visits
            self.visits[observation.action, observation.state.x_idx, observation.state.v_idx] += 1
            self.returns[observation.action, observation.state.x_idx, observation.state.v_idx] += cum_reward

            # Update Q-Value
            self.q_space[observation.action, observation.state.x_idx, observation.state.v_idx] = \
                float(self.returns[observation.action, observation.state.x_idx, observation.state.v_idx]
                      / self.returns[observation.action, observation.state.x_idx, observation.state.v_idx])

            # Policy improvement - ties will be broken by picking the first occurring action
            self.target_policy[observation.state.x_idx][observation.state.v_idx] = \
                np.argmax([self.q_space[i, observation.state.x_idx, observation.state.v_idx]
                           for i in range(len(self.action_space))])

        # Save episode results
        self.history[episode] = cum_reward
        print("Results of Episode {}".format(episode))
        print("--Cumulative Reward: {}".format(cum_reward))
        # Refresh experience
        self.experience.clear()

    def _mc_control_first_visit(self, episode: int = 0) -> None:
        """
        Monte Carlo Control algorithm - first visit
        :param episode: number of episode
        :return:
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

            self.visits[observation.action, observation.state.x_idx, observation.state.v_idx] += 1
            self.returns[observation.action, observation.state.x_idx, observation.state.v_idx] += cum_reward

            # Update Q-Value, if behavior and target policy are not equal probability distribution
            # importance sampling is required
            self.q_space[observation.action, observation.state.x_idx, observation.state.v_idx] = \
                float(self.returns[observation.action, observation.state.x_idx, observation.state.v_idx]
                      / self.returns[observation.action, observation.state.x_idx, observation.state.v_idx])

            # Policy improvement - ties will be broken by picking the first occuring action
            self.target_policy[observation.state.x_idx][observation.state.v_idx] = \
                np.argmax([self.q_space[i, observation.state.x_idx, observation.state.v_idx]
                           for i in range(len(self.action_space))])

            # Increment counter
            counter += 1

        # Save episode results
        self.history[episode] = cum_reward
        print("Results of Episode {}".format(episode))
        print("--Cumulative Reward: {}".format(cum_reward))
        # Refresh experience
        self.experience.clear()

    def choose_action(self, state: State) -> int:
        # Constant epsilon greedy policy for behavior generation
        return self.behavior_policy.eps_greedy_policy(state=state, epsilon=self.epsilon, q_space=self.q_space)
