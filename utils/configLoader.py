from typing import Dict
from agents.baseClass import Agent
from agents.mcm import MCMAgent
from agents.mdp import MDPAgent, MDP_SOLVER
from agents.tdl import TDLAgent, TDL_METHODS
import matplotlib.pyplot as plt
import gym

def load_env(config_data: Dict) -> gym.Env:
    env = gym.make(config_data["env"])
    return env

def load_model(config_data: Dict) -> Agent:
    if config_data["solver"] in ["value_iteration", "policy_iteration"]:
        agent = MDPAgent(config=config_data)
    elif "mcm" in config_data["solver"]:
        agent = MCMAgent(config=config_data)
    elif "tdl" in config_data["solver"]:
        if "tdl_sarsa" == config_data["solver"]:
            tdl_method = TDL_METHODS.SARSA
        elif "tdl_expected_sarsa" == config_data["solver"]:
            tdl_method = TDL_METHODS.EXPECTED_SARSA
        elif "tdl_q_learning" == config_data["solver"]:
            tdl_method = TDL_METHODS.Q_LEARNING
        elif "tdl_double_q_learning" == config_data["solver"]:
            tdl_method = TDL_METHODS.DOUBLE_Q_LEARNING
        else:
            raise IOError("Choose one of the available solvers!")

        agent = TDLAgent(config=config_data, tdl_method=tdl_method)
    else:
        raise IOError("Choose one of the available solvers!")

    return agent

def solve_MDP(agent: MDPAgent, plot_results: bool = False):
    if agent.solver == MDP_SOLVER.VALUE_ITERATION:
        agent.value_iteration()
        policy = agent.policy_improvement()
    elif agent.solver == MDP_SOLVER.POLICY_ITERATION:
        policy = agent.policy_iteration()
    else:
        raise IOError("Choose one of the available solvers!")

    # Visualizing the results
    if plot_results:
        plt.figure()
        plt.imshow(agent.value_space)
        plt.show()

        plt.figure()
        plt.imshow(policy)
        plt.show()
