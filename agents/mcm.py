from typing import List, Dict
import numpy as np
import copy
from env.stateSpace import State, StateSpace
import matplotlib.pyplot as plt

class Experience():
    def __init__(self, state: State, action: int, reward: float, time_step: int):
        self.state = state
        self.action = action
        self.reward = reward
        self.time_step = time_step

class MCMAgent():
    def __init__(self, config: Dict):
        # Parameter Definitions
        self.state_dim = config["state_dim"]
        self.epsilon = config["epsilon_greedy"]
        self.discount_factor = config["discount_factor"]
        self.mcm_type = config["solver"]

        # Model Definitions
        self.env = StateSpace(n=self.state_dim)

        # Space Definitions
        self.state_space_idx = self.env.get_state_space()
        self.action_space = [0, 1, 2]
        self.policy = self.init_random_policy()

        # Space of state-action values
        self.q_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))
        # Storage for averaging returns
        self.returns = np.zeros_like(self.q_space)
        self.visits = np.zeros_like(self.returns)

        # MCM specific Definitions
        self.experience: List[Experience] = list()
        self.history = dict()

        if self.mcm_type == "mcm_first_visit":
            self.mc_control = self._mc_control_first_visit
        elif self.mcm_type == "mcm_all_visit":
            self.mc_control = self._mc_control_all_visits
        else:
            raise IOError("Pick one of the available MCM Variants!")


    def init_random_policy(self) -> List[List[int]]:
        """
        Initializes a random policy
        :return:
        """
        policy = np.zeros(shape=(self.state_dim, self.state_dim))
        for i, j in self.state_space_idx:
            policy[i][j] = np.random.choice(self.action_space)
        return policy

    def add_experience(self, state_obs: List[float], action_obs: int, reward_obs: float, time_obs: int) -> None:
        """
        Adds observations to episode experience
        :param state_obs:
        :param action_obs:
        :param reward_obs:
        :param time_obs:
        :return:
        """
        self.experience.append(Experience(state=State(x=state_obs[0], v=state_obs[1]),
                                          action=action_obs,
                                          reward=reward_obs,
                                          time_step=time_obs))

    def check_first_visit(self, check_state: State, reverse_index: int) -> bool:
        first_visited = True
        check_idx = self.env.get_state_space_idx(state=check_state)
        trajectory = copy.deepcopy(self.experience)
        trajectory.pop(len(self.experience) - reverse_index - 1)
        for observation in reversed(trajectory):
            state_idx = self.env.get_state_space_idx(state=observation.state)
            if check_idx == state_idx:
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
            x_idx, v_idx = self.env.get_state_space_idx(state=observation.state)
            self.visits[observation.action, x_idx, v_idx] += 1
            self.returns[observation.action, x_idx, v_idx] += cum_reward

            # Update Q-Value
            self.q_space[observation.action, x_idx, v_idx] = \
                float(self.returns[observation.action, x_idx, v_idx] / self.returns[observation.action, x_idx, v_idx])

            # Policy improvement - ties will be broken by picking the first occuring action
            self.policy[x_idx][v_idx] = np.argmax([self.q_space[i, x_idx, v_idx] for i in range(len(self.action_space))])

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

            x_idx, v_idx = self.env.get_state_space_idx(state=observation.state)
            self.visits[observation.action, x_idx, v_idx] += 1
            self.returns[observation.action, x_idx, v_idx] += cum_reward

            # Update Q-Value
            self.q_space[observation.action, x_idx, v_idx] = \
                float(self.returns[observation.action, x_idx, v_idx] / self.returns[observation.action, x_idx, v_idx])

            # Policy improvement - ties will be broken by picking the first occuring action
            self.policy[x_idx][v_idx] = np.argmax([self.q_space[i, x_idx, v_idx] for i in range(len(self.action_space))])

            # Increment counter
            counter += 1

        # Save episode results
        self.history[episode] = cum_reward
        print("Results of Episode {}".format(episode))
        print("--Cumulative Reward: {}".format(cum_reward))
        # Refresh experience
        self.experience.clear()

    def _eps_greedy_policy(self, state: State) -> int:
        """
        Epsilon-Greedy Policy to ensure that probability of picking any action has always probability >0.
        :param state:
        :return: Int of action
        """

        # With probability of epsilon pick a random action, otherwise act greedily
        if np.random.uniform() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            x_idx, v_idx = self.env.get_state_space_idx(state=state)
            # ties will be broken by picking the first occuring action
            action = np.argmax([self.q_space[i, x_idx, v_idx] for i in range(len(self.action_space))])

        return action

    def choose_action(self, state: State) -> int:
        return self._eps_greedy_policy(state=state)

    def plot_results(self):
        plt.figure()
        plt.plot(list(self.history.keys()), list(self.history.values()))
        plt.title("Cumulative Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()