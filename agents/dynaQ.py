from typing import Dict
from agents.baseClass import Experience
from agents.tdl import TDLAgent, TDL_METHODS
from env.dynamics import TransitionModel
from env.reward import EnvReward
from enum import Enum
from env.stateSpace import State, Action
import numpy as np

class PREDICTION_MODEL_TYPE(Enum):
    EXPLICIT = 0
    LEARNED = 1


class DynaQ(TDLAgent):
    def __init__(self, config: Dict, prediction_type: PREDICTION_MODEL_TYPE):
        super().__init__(config=config,
                         tdl_method=TDL_METHODS.Q_LEARNING)

        # Dyna-Q specific attributes
        self.n_simulations = config["n_simulations"]
        self.state_action_observations: Dict = dict()
        self.reward_model = EnvReward(min_reward=-1, max_reward=0)

        if prediction_type == PREDICTION_MODEL_TYPE.EXPLICIT:
            tm = TransitionModel(_type="deterministic")
            self.prediction_model = tm.state_transition
        else:
            raise NotImplementedError

    def add_observations(self, obs: Experience) -> None:
        state_idx_tuple = obs.state.get_state_idx()

        try:
            action_list = self.state_action_observations[state_idx_tuple]
            if not len(action_list) == len(self.action_space):
                if obs.action not in action_list:
                    action_list.append(obs.action)

                self.state_action_observations[state_idx_tuple] = action_list

        except KeyError:
            self.state_action_observations[state_idx_tuple] = [obs.action]

    def simulate_n_steps(self):
        for i in range(self.n_simulations + 1):
            # Sample from experience
            sample_state_idx = np.random.choice(range(len(self.state_action_observations.keys())))
            sample_state = list(self.state_action_observations.keys())[sample_state_idx]
            sample_action = np.random.choice(self.state_action_observations[sample_state])

            # Simulate one step and get reward
            state_t0 = self.env.get_state_space_value(x_idx=sample_state[0], v_idx=sample_state[1])

            s_t1 = self.prediction_model(s_t=state_t0, a_t=sample_action)
            x_t1_idx, v_t1_idx = self.env.get_state_space_idx(observation=s_t1)
            state_t1 = State(x=s_t1[0], v=s_t1[1], x_pos=x_t1_idx, v_pos=v_t1_idx)

            reward_t1 = self.reward_model.get_reward(s=state_t1, a=sample_action)

            # Make Q-Learning Update
            self.update_func(state_t0=state_t0, action_t0=Action(a=sample_action),
                             reward=reward_t1, state_t1=state_t1)
