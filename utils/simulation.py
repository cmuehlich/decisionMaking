from typing import Dict, Union
from agents.baseClass import POLICY_LEARNING_TYPES, Experience
from agents.mdp import MDPAgent
from agents.mcm import MCMAgent
from agents.tdl import TDLAgent
from agents.dynaQ import DynaQ
from agents.mcts import MCTSAgent
import gym
from utils.visualizations import plot_results
from env.stateSpace import Action
import numpy as np


def run_mcm(config_data: Dict, agent: MCMAgent, world: gym.Env) -> None:
    for i_episode in range(config_data["env_episodes"]):
        observation = world.reset()
        current_state = agent.gen_state_from_observation(observation=observation.tolist())

        # Run simulation
        for t in range(1000):
            world.render()

            # Interact with environment
            action = agent.choose_action(current_state)
            # Get Environment feedback
            observation, reward, done, info = world.step(action)

            # Add experience
            agent.add_experience(state_obs=observation.tolist(),
                                 action_obs=action,
                                 reward_obs=reward,
                                 time_obs=t)

            # Update new state
            current_state = agent.gen_state_from_observation(observation=observation.tolist())
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        # Evaluate MCM
        agent.mc_control(episode=i_episode)

    world.close

    plot_results(x=list(agent.history.keys()), y=list(agent.history.values()),
                 meta_data={"title": "Reward per Episode",
                            "x_label": "Episodes",
                            "y_label": "Cumulative Reward"})


def run_mdp(config_data: Dict, agent: MDPAgent, world: gym.Env) -> None:
    for i_episode in range(config_data["env_episodes"]):
        observation = world.reset()
        current_state = agent.gen_state_from_observation(observation=observation.tolist())

        # Run simulation
        for t in range(1000):
            world.render()

            # Interact with environment
            action = agent.choose_action(current_state)
            # Get Environment feedback
            observation, reward, done, info = world.step(action)

            # Update new state
            current_state = agent.gen_state_from_observation(observation=observation.tolist())
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    world.close


def run_tdl(config_data: Dict, agent: TDLAgent, world: gym.Env):
    if agent.learning_type == POLICY_LEARNING_TYPES.ONLINE:
        run_tdl_online(config_data, agent, world)
    else:
        run_tdl_offline(config_data, agent, world)


def run_tdl_online(config_data: Dict, agent: TDLAgent, world: gym.Env) -> None:
    reward_episode = []
    average_rewards = []
    for i_episode in range(config_data["env_episodes"]):
        cum_reward = 0
        observation = world.reset()
        state_t0 = agent.gen_state_from_observation(observation=observation.tolist())
        action_t0 = agent.choose_action(state_t0)
        # Add experience
        agent.add_experience(state_obs=observation.tolist(),
                             action_obs=action_t0,
                             reward_obs=0,
                             time_obs=0)

        # Run simulation
        for t in range(1000):
            world.render()

            # S-A --> Interact with environment => state_t0, action_t0
            # R-S --> Get Environment feedback => state_t1, reward
            # A --> Pick next Action => action_t1
            observation, reward, done, info = world.step(action_t0)
            state_t1 = agent.gen_state_from_observation(observation=observation.tolist())
            action_t1 = agent.choose_action(state_t1)

            # Make n-step SARSA UPDATE
            # Add experience
            agent.add_experience(state_obs=observation.tolist(),
                                 action_obs=action_t1,
                                 reward_obs=reward,
                                 time_obs=t+1)

            agent.update_func(current_time_step=t)

            # Make SARSA UPDATE
            #agent.update_func(state_t0=state_t0, action_t0=Action(a=action_t0), reward=reward,
            #                  state_t1=state_t1, action_t1=Action(a=action_t1))

            # Update new state
            action_t0 = action_t1
            # state_t0 = state_t1

            # update log
            cum_reward += reward
            if done:
                reward_episode.append(cum_reward)
                if i_episode % 10 == 0:
                    print("Finished Episode: {}".format(i_episode))
                    average_rewards.append(np.mean(reward_episode))
                    reward_episode.clear()
                # print("Episode finished after {} timesteps".format(t+1))
                break
        # Free memory
        agent.clear_memory()

    world.close()

    plot_results(x=np.arange(len(average_rewards)).tolist(), y=average_rewards,
                 meta_data={"title": "Average Reward per 10/Episode",
                            "x_label": "Episodes/10",
                            "y_label": "Cumulative Reward"})


def run_tdl_offline(config_data: Dict, agent: Union[TDLAgent, DynaQ], world: gym.Env) -> None:
    reward_episode = []
    average_rewards = []
    dyna_instance = isinstance(agent, DynaQ)
    for i_episode in range(config_data["env_episodes"]):
        cum_reward = 0
        observation = world.reset()
        # Get initial state
        state_t0 = agent.gen_state_from_observation(observation=observation.tolist())
        # Choose initial action
        action_t0 = agent.choose_action(state_t0)

        # Run simulation
        for t in range(1000):
            world.render()

            # Execute action and receive feedback from the environment
            observation, reward, done, info = world.step(action_t0)
            # Get new state
            state_t1 = agent.gen_state_from_observation(observation=observation.tolist())

            # Make Q-Learning UPDATE (Maximization)
            agent.update_func(state_t0=state_t0, action_t0=Action(a=action_t0),
                              reward=reward, state_t1=state_t1)

            if dyna_instance:
                # add observation
                agent.add_observations(obs=Experience(state=state_t0, action=action_t0))

                # simulate for n times
                agent.simulate_n_steps()

            # Pick next action according to behavior policy
            action_t0 = agent.choose_action(state_t1)
            # Update new state
            state_t0 = state_t1

            # update log
            cum_reward += reward
            if done:
                reward_episode.append(cum_reward)
                if i_episode % 10 == 0:
                    print("Finished Episode: {}".format(i_episode))
                    average_rewards.append(float(np.mean(reward_episode)))
                    reward_episode.clear()
                # print("Episode finished after {} timesteps".format(t+1))
                break

    world.close()
    plot_results(x=list(np.arange(len(average_rewards))), y=average_rewards,
                 meta_data={"title": "Average Reward per 10/Episode",
                            "x_label": "Episodes/10",
                            "y_label": "Cumulative Reward"})

def run_mcts(config_data: Dict, agent: MCTSAgent, world: gym.Env) -> None:
    reward_episode = []
    average_rewards = []

    for i_episode in range(config_data["env_episodes"]):
        cum_reward = 0
        observation = world.reset()
        # Get initial state
        state_t0 = agent.gen_state_from_observation(observation=observation.tolist())
        node_t0 = agent.gen_node_from_observation(observation=observation.tolist())

        # Plan ahead
        agent.rollout(root_node=node_t0)

        # Choose initial action
        action_t0 = agent.choose_action(state_t0)

        # Run simulation
        for t in range(1000):
            world.render()

            # Execute action and receive feedback from the environment
            observation, reward, done, info = world.step(action_t0)
            # Get new state
            state_t1 = agent.gen_state_from_observation(observation=observation.tolist())
            node_t1 = agent.gen_node_from_observation(observation=observation.tolist())

            # Plan ahead
            agent.rollout(root_node=node_t1)

            # Pick next action according to behavior policy
            action_t0 = agent.choose_action(state_t1)

            # update log
            cum_reward += reward
            if done:
                reward_episode.append(cum_reward)
                if i_episode % 10 == 0:
                    print("Finished Episode: {}".format(i_episode))
                    average_rewards.append(float(np.mean(reward_episode)))
                    reward_episode.clear()
                # print("Episode finished after {} time steps".format(t+1))
                break

    world.close()
    plot_results(x=list(np.arange(len(average_rewards))), y=average_rewards,
                 meta_data={"title": "Average Reward per 10/Episode",
                            "x_label": "Episodes/10",
                            "y_label": "Cumulative Reward"})
    agent.plot_graph(root_node=node_t0)