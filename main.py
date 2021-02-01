import gym
import yaml
from agents.mdp import MDPAgent
import matplotlib.pyplot as plt
import os

def main(config):
    # Load Environment
    env = gym.make("MountainCar-v0")
    observer = []

    # Initialize Decision Maker
    agent = MDPAgent(config=config)

    # Compute policy
    if config["solver"] == "value_iteration":
        agent.value_iteration()
        policy = agent.policy_improvement()
    elif config["solver"] == "policy_iteration":
        policy = agent.policy_iteration()
    else:
        raise IOError("Choose one of the available solvers!")

    # Visualizing the results
    if config["plot_value_space"]:
        plt.imshow(agent.value_space)
        plt.show()
    if config["plot_policy_space"]:
        plt.imshow(policy)
        plt.show()

    for i_episode in range(config["env_episodes"]):
        observation = env.reset()
        observer.append(observation.tolist())
        current_state = observer[-1]
        for t in range(10000):
            env.render()
            action = agent.get_optimal_action(policy, current_state)
            observation, reward, done, info = env.step(action)
            observer.append(observation.tolist())
            current_state = observer[-1]
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close

if __name__ == '__main__':
    with open(os.getcwd() + "/config/config.yaml", "r") as file:
        config_data = yaml.full_load(file)
    main(config=config_data)
