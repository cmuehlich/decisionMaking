from typing import Dict, Union, Tuple
from agents.baseClass import POLICY_LEARNING_TYPES, POLICY_TYPES, Agent, Node
from agents.dynaQ import PREDICTION_MODEL_TYPE
from env.dynamics import TransitionModel
from env.reward import EnvReward
from env.stateSpace import State, STATE_SPACE_TYPE
import numpy as np
import queue
import networkx as nx


class MCTSAgent(Agent):
    def __init__(self, config: Dict, prediction_type: PREDICTION_MODEL_TYPE):
        if prediction_type == PREDICTION_MODEL_TYPE.EXPLICIT:
            tm = TransitionModel(_type="deterministic")
        else:
            raise NotImplementedError

        super().__init__(config=config,
                         state_space_type=STATE_SPACE_TYPE.DISCRETE,
                         reward_model=EnvReward(min_reward=-1, max_reward=0),
                         system_dynamics=tm,
                         learning_type=POLICY_LEARNING_TYPES.OFFLINE,
                         initial_target_policy=POLICY_TYPES.EPS_GREEDY,
                         initial_behavior_policy=POLICY_TYPES.EPS_GREEDY)

        # State Transitions
        self.prediction_model = self.transition_model.state_transition
        # Root node of tree
        self.root: Union[Node, None] = None
        self.ucb_alpha = config["ucb_alpha"]
        self.n_iterations = config["mcts_iterations"]

        # State Action Value Space
        self.q_space = np.random.random(size=(len(self.action_space), self.state_dim, self.state_dim))

        # LIFO queue
        self.frontier = queue.LifoQueue()
        self.action_trajectory = queue.LifoQueue()

        # Reward Collector
        self.reward_collection = 0

    def set_root_node(self, node: Node) -> None:
        if self.root is None:
            self.root = node
        else:
            raise IOError("Root Node can't be changed during runtime.")

    def find_node(self, ref_state: State) -> Union[Node, None]:
        search_tree = queue.Queue()

        # If tree has not expanded yet, return none
        if not self.root.expanded:
            return None

        check_node = self.root
        found_node: Union[Node, None] = None

        while found_node is None or not search_tree.empty():
            found_node, search_tree = self.tree_traversal(ref_node=check_node, ref_state=ref_state,
                                                          search_tree=search_tree)
            check_node = search_tree.get()

        return found_node

    def tree_traversal(self, ref_node: Node, ref_state: State, search_tree: queue.LifoQueue) \
            -> Tuple[Union[Node, None], queue.LifoQueue]:
        for action_id in self.action_space:
            nodes_list = ref_node.successor_nodes[action_id]
            if len(nodes_list) == 0:
                continue

            for node in nodes_list:
                search_tree.put(node)
                if node.state.get_state_idx() == ref_state.get_state_idx():
                    return node, search_tree

            return None, search_tree

    def rollout(self, root_node: Node):
        for n in range(self.n_iterations):
            self.reward_collection = 0

            # Get leaf node
            leaf_node = self.selection_process(node=root_node)

            # Expand tree
            child_node = self.expansion_process(leaf_node=leaf_node)

            # Make rollout
            self.simulation_process(child_node=child_node)

            # Backup tree
            self.backup_process()

    def compute_ucb(self, node: Node) -> Node:
        best_action = np.argmax([self.q_space[action_id][node.state.x_idx][node.state.v_idx] +
                                 self.ucb_alpha * np.sqrt(2*np.log(node.total_visits) / node.action_visits[action_id])
                                 for action_id in self.action_space])

        # Get successor node of best action
        next_node = node.successor_nodes[int(best_action)][0]

        self.reward_collection += self.reward_model.get_reward(s=next_node.state, a=int(best_action))
        self.action_trajectory.put(best_action)
        return next_node

    def selection_process(self, node: Node) -> Node:
        # Traverse node until leaf node has been found
        leaf_node = None
        found_leaf_node = False

        traversed_node = node

        while not found_leaf_node:
            leaf_node = traversed_node
            self.frontier.put(traversed_node)

            # If node has been visited, check if all actions have been taken at least once
            available_actions = [self.action_space[i] for i in range(len(self.action_space))
                                 if traversed_node.action_visits[i] == 0]
            if node.total_visits == np.inf or len(available_actions) > 0:
                found_leaf_node = True
            else:
                # if node has not been leaf node, traverse one step more
                # pick according to UCB
                traversed_node = self.compute_ucb(node=traversed_node)

        return leaf_node

    def expansion_process(self, leaf_node: Node) -> Node:
        # if not all actions were explored choose randomly from those
        available_actions = [self.action_space[i] for i in range(len(self.action_space))
                             if leaf_node.action_visits[i] == 0]
        action = np.random.choice(available_actions)

        # Simulate next state
        next_state_values = self.prediction_model(s_t=leaf_node.state, a_t=action)
        next_state_idx = self.env.get_state_space_idx(observation=next_state_values)
        next_state = State(x=next_state_values[0], v=next_state_values[1],
                           x_pos=next_state_idx[0], v_pos=next_state_idx[1])

        self.reward_collection += self.reward_model.get_reward(s=next_state, a=action)

        # Add node to tree
        child_node = Node(state=next_state, action_space=self.action_space)
        leaf_node.add_successor(node=child_node, action=action)

        # Update node
        if leaf_node.total_visits == np.inf:
            leaf_node.total_visits = 1
        else:
            leaf_node.total_visits += 1

        leaf_node.action_visits[action] += 1
        self.action_trajectory.put(action)

        return child_node

    def simulation_process(self, child_node: Node) -> None:
        # From child node onwards sample randomly for n time steps or until terminal state and collect rewards
        current_state = child_node.state

        for t in range(self.episode_duration):
            # Sample action randomly
            action = np.random.choice(self.action_space)
            # Simulate next state
            next_state_values = self.prediction_model(s_t=current_state, a_t=action)
            next_state_idx = self.env.get_state_space_idx(observation=next_state_values)
            next_state = State(x=next_state_values[0], v=next_state_values[1],
                               x_pos=next_state_idx[0], v_pos=next_state_idx[1])

            # Add reward
            self.reward_collection += self.reward_model.get_reward(s=next_state, a=action)

            # check if terminal state
            if self.env.is_terminal(state=next_state):
                break
            else:
                current_state = next_state

    def backup_process(self) -> None:
        assert self.action_trajectory.qsize() == self.frontier.qsize()

        while not self.frontier.empty():
            picked_action = self.action_trajectory.get()
            visited_node = self.frontier.get()

            # Update Q-Values
            action_rewards = visited_node.action_rewards[picked_action]
            action_rewards.append(self.reward_collection)
            visited_node.action_rewards[picked_action] = action_rewards
            self.q_space[picked_action][visited_node.state.x_idx][visited_node.state.v_idx] = \
                np.sum(action_rewards) / visited_node.action_visits[picked_action]

    def choose_action(self, state: State) -> int:
        return self.target_policy.eps_greedy_policy(state=state, epsilon=self.epsilon, q_space=self.q_space)

    def plot_graph(self, root_node: Node):
        G = nx.DiGraph()
        layout = dict()
        layout[root_node.uuid] = root_node.state.get_state_value()
        G.add_node(root_node.uuid)
        search_tree = queue.Queue()
        search_tree.put(root_node)


        while not search_tree.empty():
            top_node = search_tree.get()
            for action_id in self.action_space:
                nodes_list = top_node.successor_nodes[action_id]
                if len(nodes_list) == 0:
                    continue
                for node in nodes_list:
                    search_tree.put(node)
                    G.add_node(node.uuid)
                    layout[node.uuid] = node.state.get_state_value()
                    G.add_edge(top_node.uuid, node.uuid)


        nx.draw(G, pos=layout, node_size=5)