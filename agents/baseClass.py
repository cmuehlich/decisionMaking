from typing import Dict, Union, List, Tuple
from env.stateSpace import State, StateSpace
from env.reward import RewardModel
from env.dynamics import TransitionModel
import numpy as np
from enum import Enum
import abc
import uuid


class POLICY_LEARNING_TYPES(Enum):
    OFFLINE = 0
    ONLINE = 1


class POLICY_TYPES(Enum):
    RANDOM = 0
    EPS_GREEDY = 1
    UCB = 2


class Node():
    def __init__(self, state: State, action_space: List[int]):
        self.state = state
        self.action_space = action_space

        self.uuid = uuid.uuid4()
        self.expanded = False

        # List of successor nodes stored as list of list, containing one set of nodes for each action
        self.successor_nodes: List[List[Node]] = [list() for i in range(len(self.action_space))]

        self.total_visits = np.inf
        self.action_visits = np.zeros(len(self.action_space))
        self.action_rewards = [list() for i in range(len(self.action_space))]

    def add_successor(self, node, action: int) -> None:
        if node not in self.successor_nodes[action]:
            nodes_list = self.successor_nodes[action]
            nodes_list.append(node)
            self.successor_nodes[action] = nodes_list


class Policy:
    def __init__(self, policy_type: POLICY_TYPES):
        self.policy_type = policy_type
        self.state_space_idx: np.ndarray = np.ndarray
        self.state_dim: int = 0
        self.action_space: List[int] = list()

    def init_policy(self, state_dim: int, state_space_idx: List[Tuple[int, int]], action_space: List[int]) -> Union[np.ndarray, None]:
        self.state_space_idx = state_space_idx
        self.state_dim = state_dim
        self.action_space = action_space

        if self.policy_type == POLICY_TYPES.RANDOM:
            return self._random_policy()
        else:
            return None

    def _random_policy(self) -> np.ndarray:
        """
        Initializes a random policy
        :return:
        """
        policy = np.zeros(shape=(self.state_dim, self.state_dim))
        for i, j in self.state_space_idx:
            policy[i][j] = np.random.choice(self.action_space)
        return policy

    def eps_greedy_policy(self, state: State, epsilon: float, q_space: np.ndarray) -> int:
        """
        Epsilon-Greedy Policy to ensure that probability of picking any action has always probability >0.
        :param state: current State
        :param epsilon: probability threshold
        :param q_space: State-Action Values
        :return: Index of action
        """

        # With probability of epsilon pick a random action, otherwise act greedily
        if np.random.uniform() <= epsilon:
            action = np.random.choice(self.action_space)
        else:
            # ties will be broken by picking the first occuring action
            action = np.argmax([q_space[i, state.x_idx, state.v_idx] for i in range(len(self.action_space))])

        return action

    def get_eps_greedy_probabilities(self, epsilon: float) -> Tuple[float, float]:
        """
        Computes the probabilities of taking action according to greedy policy
        :param epsilon: probability threshold
        :return: Tuple (prob_of_greedy_action, prob_of_exploration)
        """
        prob_of_greedy_action = 1 - epsilon + (epsilon / len(self.action_space))
        prob_of_exploration = epsilon / len(self.action_space)
        return prob_of_greedy_action,prob_of_exploration

    def ucb_policy(self, node: Node, beta: float, q_space: np.ndarray) -> int:
        pass


class Experience:
    def __init__(self, state: State, action: int, reward: float = 0, time_step: int = 0):
        self.state = state
        self.action = action
        self.reward = reward
        self.time_step = time_step


class Agent(abc.ABC):
    def __init__(self, config: Dict, reward_model: Union[RewardModel, None],
                 system_dynamics: Union[TransitionModel, None], learning_type: Union[POLICY_LEARNING_TYPES, None],
                 initial_target_policy: Union[POLICY_TYPES, None], initial_behavior_policy: Union[POLICY_TYPES, None]):
        # Parameter Definitions
        self.state_dim = config["state_dim"]
        self.epsilon = config["epsilon_greedy"]
        self.discount_factor = config["discount_factor"]

        # Experience Collector
        self.experience: List[Experience] = list()
        if config["n_step"] < 1 or not isinstance(config["n_step"], int):
            raise IOError("N-Step Config Parameter needs to be of type integer and >= 1")
        self.n_step = config["n_step"]
        self.episode_duration = config["episode_duration"]

        # Model definitions
        self.learning_type = learning_type
        self.env = StateSpace(n=self.state_dim)
        self.reward_model = reward_model
        self.transition_model = system_dynamics

        # State Space Configuration
        self.state_space_idx, self.state_list = self.env.get_state_space()

        # Action Space Configuration -> here: only Mountain Car
        self.action_space = [0, 1, 2]

        # Decision Making Configuration
        self.target_policy = None
        self.behavior_policy = None
        self.init_policies(initial_target_policy=initial_target_policy,
                           initial_behavior_policy=initial_behavior_policy)

        if self.learning_type == POLICY_LEARNING_TYPES.OFFLINE and self.behavior_policy is None:
            raise IOError("For Off-policy learning a behavior policy needs to be provided!")

        if self.learning_type is None and (self.target_policy is not None or self.behavior_policy is not None):
            raise IOError("Specify the desired policy learning method")

    def init_policies(self, initial_target_policy: Union[POLICY_TYPES, None], initial_behavior_policy: Union[POLICY_TYPES, None]) -> None:
        random_policy = None
        rd_policy = None
        greedy_policy = None

        # Random policy
        if POLICY_TYPES.RANDOM in [initial_target_policy, initial_behavior_policy]:
            random_policy = Policy(policy_type=initial_target_policy)
            rd_policy = random_policy.init_policy(state_dim=self.state_dim,
                                                  state_space_idx=self.state_space_idx,
                                                  action_space=self.action_space)
        # Epsilon-Greedy policy
        if POLICY_TYPES.EPS_GREEDY in [initial_target_policy, initial_behavior_policy]:
            greedy_policy = Policy(policy_type=initial_target_policy)
            greedy_policy.init_policy(state_dim=self.state_dim,
                                      state_space_idx=self.state_space_idx,
                                      action_space=self.action_space)

        if initial_target_policy == POLICY_TYPES.RANDOM:
            self.target_policy = rd_policy
        elif initial_target_policy == POLICY_TYPES.EPS_GREEDY:
            self.target_policy = greedy_policy

        if initial_behavior_policy == POLICY_TYPES.RANDOM:
            self.behavior_policy = rd_policy
        elif initial_behavior_policy == POLICY_TYPES.EPS_GREEDY:
            self.behavior_policy = greedy_policy

    def gen_state_from_observation(self, observation: List[float]) -> State:
        """
        Generates a state object for the environment observation
        :param observation:
        :return:
        """
        x_idx, v_idx = self.env.get_state_space_idx(observation=observation)
        return State(x=observation[0], v=observation[1], x_pos=x_idx, v_pos=v_idx)

    def gen_node_from_observation(self, observation: List[float]) -> Node:
        """
        Generates a node object for the environment observation
        :param observation:
        :return:
        """
        x_idx, v_idx = self.env.get_state_space_idx(observation=observation)
        state = State(x=observation[0], v=observation[1], x_pos=x_idx, v_pos=v_idx)
        return Node(state=state, action_space=self.action_space)

    def add_experience(self, state_obs: List[float], action_obs: int, reward_obs: float, time_obs: int) -> None:
        """
        Adds observations to episode experience
        :param state_obs:
        :param action_obs:
        :param reward_obs:
        :param time_obs:
        :return:
        """
        observed_state = self.gen_state_from_observation(observation=state_obs)
        self.experience.append(Experience(state=observed_state,
                                          action=action_obs,
                                          reward=reward_obs,
                                          time_step=time_obs))

    def clear_memory(self) -> None:
        """
        Resets the memory storage of past state-action-reward trajectories.
        :return:
        """
        self.experience.clear()

    @abc.abstractmethod
    def choose_action(self, state: State) -> int:
        """
        Placeholder for decision making
        :return: Action id of Action space
        """
        raise NotImplementedError
