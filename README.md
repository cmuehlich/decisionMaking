# Decision Making

This repo contains basic implementations of decision making algorithms based on [OpenAI](https://gym.openai.com/envs/#classic_control) Gym Environments. All variants will be implemented for the cart-pole environment for comparison purposes.

Current implementations:

1) **Deterministic Markov Decision Process**
* *Value Iteration*
* *Policy Iteration*

Open ToDo list: Stochastic MDP, Reinforcement Learning Variants

# Getting started

First make sure you have all required packages installed. After cloning the repo, change to the directory and run

`pip install -r requirements.txt`

If you work with anaconda, make sure to activate your environment before installing.

In the /config directory you find a config.yaml file where you can adjust some parameters like the discount factor or the mdp solver.

Then change to the main directory and run

`python main.py`

After the MDP is solved three things will open up. The first plot is showing a color coded plot of the discretized value space. For more information about the state space, check the cart-pole documentation. The second plot shows the corresponding policy space. Each cell's color indicating which of the three available actions the car should execute. And finally a window will open up displaying the solution to the simulated environment. 
