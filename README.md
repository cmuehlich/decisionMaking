# Decision Making

This repo contains basic implementations of decision making algorithms based on [OpenAI](https://gym.openai.com/envs/#classic_control) Gym Environments. All variants will be implemented for the mountain-car environment for comparison purposes.

Current implementations:

1) **Deterministic Markov Decision Process**
* *Value Iteration*
* *Policy Iteration*

2) **Monte Carlo Reinforcement Learning**

* *Monte Carlo First-Visits - On-Policy Learning*
* *Monte Carlo All-Visits - On-Policy Learning*

3) **Temporal Difference Learning - TD(0)**

* *SARSA - On-Policy learning*
* *Expected SARSA - Off-Policy learning*
* *Q-Learning - Off-Policy Learning*
* *Double-Q-Learning - Off-Policy Learning* 

4) **Integrated Planning & Learning** 

* *Dyna-Q with explicit environment model*
* *Monte Carlo Tree Search with explicit environment model*

5) **Value-Based Function Approximation**

* *Semi-Gradient SARSA - Linear Function Approximation with code tiling*

6) **Policy-Based Function Approximation - Policy Gradient Methods**

* *Monte Carlo REINFORCE algorithm*
* *Monte Carlo REINFORCE algorithm with baseline*
* *Actor-Critic method*

Open ToDo list: Stochastic MDP, Importance Sampling for Off-Policy, Learned Models for IPL

# Getting started

First make sure you have all required packages installed. After cloning the repo, change to the directory and run

`pip install -r requirements.txt`

If you work with anaconda, make sure to activate your environment before installing.

In the /config directory you find a *config.yaml* file where you can adjust some parameters like the discount factor or the mdp solver.

Then change to the main directory and run

`python main.py`

#### MDPs 

tbd

#### MCMs

tbd

#### TDLs

tbd

#### IPL Methods

tbd

# Configuration

tbd