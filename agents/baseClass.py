from typing import Dict, Union, List, Tuple
from env.stateSpace import State, DiscreteStateSpace, ContinuousStateSpace, STATE_SPACE_TYPE, ACTION_SPACE_TYPE
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
    EPS_GREEDY_CONT = 2
    FUNC = 3


class Node:
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
        self.state_space_idx: Union[np.ndarray, None] = None
        self.state_dim: Union[int, None] = None
        self.action_space: Union[List[int], None] = None

    def init_policy(self, state_dim: Union[int, None], state_space_idx: Union[List[Tuple[int, int]], None] = None,
                    action_space: ACTION_SPACE_TYPE = ACTION_SPACE_TYPE.DISCRETE) -> None:
        if self.policy_type == POLICY_TYPES.EPS_GREEDY:
            self.state_space_idx = state_space_idx
            self.state_dim = state_dim
            self.action_space = action_space
        else:
            self.action_space = action_space.value

    def get_action(self, state: Union[State, None] = None, epsilon: float = 0.1,
                   q_space: Union[np.ndarray, None] = None, feature_vector: Union[np.ndarray, None] = None) -> int:
        if self.policy_type == POLICY_TYPES.EPS_GREEDY:
            return self._eps_greedy_policy(state=state, epsilon=epsilon, q_space=q_space)
        elif self.policy_type == POLICY_TYPES.EPS_GREEDY_CONT:
            return self._eps_greedy_continuous_state_policy(epsilon=epsilon, q_space=q_space,
                                                            feature_vector=feature_vector)
        elif self.policy_type == POLICY_TYPES.FUNC:
            action_probs = self.action_preference_distribution(q_space=q_space, feature_vector=feature_vector)
            return int(np.argmax(action_probs))
        elif self.policy_type == POLICY_TYPES.RANDOM:
            return np.random.choice(self.action_space)

    def _eps_greedy_policy(self, state: State, epsilon: float, q_space: np.ndarray) -> int:
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
            # ties will be broken by picking the first occurring action
            action = np.argmax([q_space[i, state.x_idx, state.v_idx] for i in range(len(self.action_space))])

        return action

    def _eps_greedy_continuous_state_policy(self, epsilon: float, q_space: np.ndarray, feature_vector: np.ndarray) -> int:
        """
        Epsilon-Greedy Policy for continous state space.
        :param state:
        :param epsilon:
        :param q_space:
        :param feature_vector:
        :return:
        """
        # With probability of epsilon pick a random action, otherwise act greedily
        if np.random.uniform() <= epsilon:
            action = np.random.choice(self.action_space)
        else:
            # ties will be broken by picking the first occurring action
            action = np.argmax([q_space[i].dot(feature_vector) for i in range(len(self.action_space))])

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

    def action_preference_distribution(self, q_space: np.ndarray, feature_vector: np.ndarray) -> List[float]:
        preference_values = [q_space[i].dot(feature_vector) for i in range(len(self.action_space))]

        eq_check = [1 for i in range(len(preference_values) - 1) if preference_values[i] == preference_values[i+1]]
        if len(eq_check) == len(self.action_space) - 1:
            action_probs = [0 for i in range(len(self.action_space))]
            action_probs[np.random.choice(self.action_space)] = 1
        else:
            norm = 0
            probs = list()
            for val in preference_values:
                exp_val = np.exp(val)
                probs.append(exp_val)
                norm += exp_val

            action_probs = [i/norm for i in probs]
        return action_probs


class Experience:
    def __init__(self, state: State, action: int, reward: float = 0, time_step: int = 0):
        self.state = state
        self.action = action
        self.reward = reward
        self.time_step = time_step


class Agent(abc.ABC):
    def __init__(self, config: Dict, state_space_type: STATE_SPACE_TYPE,
                 action_space_type: ACTION_SPACE_TYPE = ACTION_SPACE_TYPE.DISCRETE,
                 reward_model: Union[RewardModel, None] = None,
                 system_dynamics: Union[TransitionModel, None] = None,
                 learning_type: Union[POLICY_LEARNING_TYPES, None] = None,
                 initial_target_policy: Union[POLICY_TYPES, None] = None,
                 initial_behavior_policy: Union[POLICY_TYPES, None] = None):
        # Parameter Definitions
        self.epsilon = config["epsilon_greedy"]
        self.discount_factor = config["discount_factor"]
        self.learning_rate = config["learning_rate"]
        assert 0 <= self.discount_factor <= 1

        # Experience Collector
        self.experience: List[Experience] = list()
        if config["n_step"] < 1 or not isinstance(config["n_step"], int):
            raise IOError("N-Step Config Parameter needs to be of type integer and >= 1")
        self.n_step = config["n_step"]
        self.episode_duration = config["episode_duration"]

        # Model definitions
        self.learning_type = learning_type
        self.state_space_type = state_space_type
        if state_space_type == state_space_type.DISCRETE:
            self.state_dim = config["state_dim"]
            self.env = DiscreteStateSpace(n=self.state_dim)
            # State Space Configuration
            self.state_space_idx, self.state_list = self.env.get_state_space()
        else:
            self.state_dim = None
            self.state_space_idx = None
            self.env = ContinuousStateSpace()

        self.reward_model = reward_model
        self.transition_model = system_dynamics

        # Action Space Configuration -> here: only Mountain Car
        self.action_space_type = action_space_type
        self.action_space = action_space_type.value

        # Decision Making Configuration
        self.target_policy = None
        self.behavior_policy = None
        self.init_policies(initial_target_policy=initial_target_policy,
                           initial_behavior_policy=initial_behavior_policy)

        if self.learning_type == POLICY_LEARNING_TYPES.OFFLINE and self.behavior_policy is None:
            raise IOError("For Off-policy learning a behavior policy needs to be provided!")

        if self.learning_type is None and (self.target_policy is not None or self.behavior_policy is not None):
            raise IOError("Specify the desired policy learning method")

    def init_policies(self, initial_target_policy: Union[POLICY_TYPES, None],
                      initial_behavior_policy: Union[POLICY_TYPES, None]) -> None:

        self.behavior_policy = Policy(policy_type=initial_behavior_policy)
        self.behavior_policy.init_policy(state_dim=self.state_dim,
                                         state_space_idx=self.state_space_idx,
                                         action_space=self.action_space_type)

        self.target_policy = Policy(policy_type=initial_target_policy)
        self.target_policy.init_policy(state_dim=self.state_dim,
                                       state_space_idx=self.state_space_idx,
                                       action_space=self.action_space_type)

    def gen_state_from_observation(self, observation: List[float]) -> State:
        """
        Generates a state object for the environment observation
        :param observation:
        :return:
        """
        if self.state_space_type == STATE_SPACE_TYPE.DISCRETE:
            x_idx, v_idx = self.env.get_state_space_idx(observation=observation)
        else:
            x_idx, v_idx = None, None
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
