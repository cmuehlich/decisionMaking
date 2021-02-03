import gym
import yaml
import numpy as np
from env.stateSpace import State
from agents.mdp import MDPAgent
from agents.mcm import MCMAgent
import matplotlib.pyplot as plt
import os

def main(config):
    # Load Environment
    env = gym.make("MountainCar-v0")

    # # Initialize Decision Maker and compute policy
    if config["solver"] == "value_iteration":
        agent = MDPAgent(config=config)
        agent.value_iteration()
        policy = agent.policy_improvement()
    elif config["solver"] == "policy_iteration":
        agent = MDPAgent(config=config)
        policy = agent.policy_iteration()
    elif "mcm" in config["solver"]:
        agent = MCMAgent(config=config)
    else:
        raise IOError("Choose one of the available solvers!")

    # Visualizing the results
    if config["solver"] in ["value_iteration", "policy_iteration"]:
        if config["plot_value_space"]:
            plt.imshow(agent.value_space)
            plt.show()
        if config["plot_policy_space"]:
            plt.imshow(policy)
            plt.show()

    for i_episode in range(config["env_episodes"]):
        observation = env.reset()
        state_observation = observation.tolist()
        current_state = State(x=state_observation[0], v=state_observation[1])

        # Run simulation
        for t in range(1000):
            env.render()

            # Interact with environment
            action = agent.choose_action(current_state)
            # Get Environment feedback
            observation, reward, done, info = env.step(action)

            # Add experience
            agent.add_experience(state_obs=state_observation,
                                 action_obs=action,
                                 reward_obs=reward,
                                 time_obs=t)

            # Update new state
            state_observation = observation.tolist()
            current_state = State(x=state_observation[0], v=state_observation[1])
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        if config["solver"] in ["value_iteration", "policy_iteration"]:
            continue
        # Evaluate MCM
        agent.mc_control(episode=i_episode)

    env.close

    if config["solver"] == "mcm":
        agent.plot_results()

if __name__ == '__main__':
    with open(os.getcwd() + "/config/config.yaml", "r") as file:
        config_data = yaml.full_load(file)
    main(config=config_data)
