from typing import Dict, TYPE_CHECKING
import yaml
from utils.configLoader import load_model, solve_MDP, load_env
from utils.simulation import run_mcm, run_mdp, run_tdl, run_mcts
from agents.mdp import MDPAgent
from agents.mcm import MCMAgent
from agents.tdl import TDLAgent
from agents.mcts import MCTSAgent
import os

def main(config: Dict):
    # Load Environment
    env = load_env(config_data=config)

    # Initialize Decision Maker and compute policy
    agent = load_model(config_data=config)

    if isinstance(agent, MDPAgent):
        print("## Solving MDP ...")
        solve_MDP(agent=agent, plot_results=config["plot_results"])
        print("## Start running episodes")
        run_mdp(config_data=config, agent=agent, world=env)
    elif isinstance(agent, MCMAgent):
        print("## Start running episodes")
        run_mcm(config_data=config, agent=agent, world=env)
    elif isinstance(agent, TDLAgent):
        print("## Start running episodes")
        run_tdl(config_data=config, agent=agent, world=env)
    elif isinstance(agent, MCTSAgent):
        print("## Start running episodes")
        run_mcts(config_data=config, agent=agent, world=env)


if __name__ == '__main__':
    with open(os.getcwd() + "/config/config.yaml", "r") as file:
        config_data = yaml.full_load(file)
    main(config=config_data)
