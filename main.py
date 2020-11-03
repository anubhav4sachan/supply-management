# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import os
import time

from sa_env import State, Action
from environment import Environment
from policy import BaselinePolicy
from utils import VisualPlots

if __name__ == '__main__':
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    envt = Environment()
    policy = BaselinePolicy(10.0, 40.0, [2, 2, 2, 2], [12, 20, 11, 15])

    num_episodes = 100
    ep_rewards = []

    for i in range(num_episodes):
        envt.reset()

        state = envt.initial_state()
        transitions = []

        for _ in range(envt.eps):
            action = policy.choose_action(state)
            state, rewards, _ = envt.step(state, action)
            transitions.append([state, action, rewards])

        ep_rewards.append(sum(np.array(transitions).T[2]))

    image_dpi = 300
    plt.figure(figsize=(16, 5), dpi=image_dpi)
    plt.plot(range(len(ep_rewards)), ep_rewards)

    graphs_dir = "graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        
    plt.style.use('seaborn-whitegrid')
    plt.savefig('{}/{}-1.png'.format(graphs_dir, current_time), dpi=image_dpi)

    envt = Environment()
    
    transitions = []
    for _ in range(envt.eps):
        action = policy.choose_action(state)
        state, rewards, _ = envt.step(state, action)
        transitions.append([state, action, rewards])

    pl = VisualPlots(graphs_dir, current_time, envt, image_dpi)
    pl.plot_transition((np.array(transitions)))
    
    print("Simulation Complete, please check the graphs folder.")
