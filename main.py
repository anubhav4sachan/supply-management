# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import os
import time
from ax import optimize

from sa_env import State, Action
from environment import Environment
from policy import BaselinePolicy
from utils import VisualPlots


def eps_sim(envt, policy):
    state = envt.initial_state()
    transitions = []

    for _ in range(envt.eps):
        action = policy.choose_action(state)
        state, rewards, _ = envt.step(state, action)
        transitions.append([state, action, rewards])

    return transitions


def sim(envt, policy, num_episodes):
    r_array = []
    for episode in range(num_episodes):
        envt.reset()
        r_array.append(sum(np.array(eps_sim(envt, policy)).T[2]))
    return r_array


def f_policy(p):
    policy = BaselinePolicy(
        p['factory_s'],
        p['factory_Q'],
        [p['w1_s'], p['w2_s'], p['w3_s'], p['w4_s']],
        [p['w1_Q'], p['w2_Q'], p['w3_Q'], p['w4_Q']]
    )
    return np.mean(sim(envt, policy, num_episodes=30))


if __name__ == '__main__':
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    envt = Environment()
    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {"name": "factory_s", "type": "range", "bounds": [0.0, 30.0], },
            {"name": "factory_Q", "type": "range", "bounds": [0.0, 30.0], },
            {"name": "w1_s", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w1_Q", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w2_s", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w2_Q", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w3_s", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w3_Q", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w4_s", "type": "range", "bounds": [0.0, 20.0], },
            {"name": "w4_Q", "type": "range", "bounds": [0.0, 20.0], },
        ],
        evaluation_function=f_policy,
        minimize=False,
        total_trials=200,
    )
    
    print("Best Params:", best_parameters)
    print("Best Value:", best_values)

    num_eps = 100
    envt = Environment()
    policy = BaselinePolicy(0.0, 25.0, [5, 5, 5, 5], [5, 5, 10, 20])
    ep_rewards = sim(envt, policy, num_episodes=num_eps)

    transitions = eps_sim(envt, policy)

    graphs_dir = "graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    pl = VisualPlots(graphs_dir, current_time, envt, 300)
    pl.plot_transition((np.array(transitions)))

    print("Simulation Complete, please check the graphs folder.")
