#### Blueyonder Crystal Ball 2020 Hackathon Challenge

# Demonstrate that a distributed Supply Chain Problem can be managed by AI Agents

##### [Hackathon URL](https://blueyonder.mentormind.in/hackathons/demonstrate-that-a-distributed-supply-chain-problem-can-be-managed-by-co-operating-ai-agents-e48c55de-7a03-45d7-b903-5584e97c9471) | Reinforcement Learning | Deep Deterministic Policy Gradient Algorithm

### Components:
The problem statement was divided into three components -:

- __Component 1__: Create a supply chain environment to train AI agents that play the supply chain management.
    - [[pdf](docs/c1.pdf)] [[environment.py](environment.py)] Environment formulation, to set up a working environment for the agent's seamless interaction. 

- __Component 2__: Create algorithm agents that follow a logic to manage item locations.
  - [[pdf](docs/c2.pdf)] [[policy.py](policy.py)] Created baseline (s, Q) policy, and used Bayesian Optimization from [Facebook Ax Platform](https://ax.dev/) to determine the best set of parameters (such as safe stock levels at different warehouses, production level at the factory, etc.) for the supply chain problem. 

- __Component 3__: Use Machine Learning (Reinforcement Learning) to build a model that makes the same agent decisions.
  - [[pdf](docs/c3.pdf)] [[DDPG.ipynb](DDPG.ipynb)] Using [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) Algorithm, we focused on determining the optimal policy, to maximize the return (profit for the supplier). 

### Notes:
- _For DDPG, please run the IPython Notebook. For Bayesian Optimization, run `main.py`._