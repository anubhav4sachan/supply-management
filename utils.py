#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import pyplot as plt
import numpy as np


class VisualPlots:
    def __init__(self, grdr, curr_time, envt, image_dpi):
        self.tt = curr_time
        self.graphs_dir = grdr
        self.img_dpi = image_dpi
        self.env = envt

    def setup_plot(self, plt_total, n, lbl):
        plt.subplot(plt_total, 1, n)
        plt.ylabel(lbl)
        plt.tick_params(axis='x', which='both', bottom=True, top=True)

    def plot_transition(self, transitions):
        state_arr, action_arr, reward_arr = (transitions.T[0], transitions.T[1], transitions.T[2])
        plt_total = 5
        matplotlib.rcParams['lines.linewidth'] = 1

        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 10), dpi=self.img_dpi)

        self.setup_plot(plt_total, 1, "Production")
        plt.plot(range(self.env.eps), list(map(lambda a: a.production_level, action_arr)), c='purple', alpha=0.5)

        self.setup_plot(plt_total, 2, "Stock,\n WH2")
        plt.plot(range(self.env.eps), list(map(lambda s: s.warehouse_stock[1], state_arr)), c='brown', alpha=0.5)

        self.setup_plot(plt_total, 3, "Shipment,\n WH2")
        plt.plot(range(self.env.eps), list(map(lambda a: a.shippings[1], action_arr)), c='green', alpha=0.5)

        self.setup_plot(plt_total, 4, "Individual\nProfit")
        plt.plot(range(self.env.eps), reward_arr, c='orange', alpha=0.9, linewidth=1)

        plt.subplot(plt_total, 1, 5)
        plt.ylabel("Cumulative\nprofit")
        plt.ylim(-10000, 10000)
        plt.xlabel("Time")
        print('Profit Array at {}\n{}'.format(self.tt, np.cumsum(reward_arr)))
        plt.plot(range(self.env.eps), np.cumsum(reward_arr), c='red', alpha=0.9, linewidth=1.5)
        
        plt.savefig('{}/{}-2.png'.format(self.graphs_dir, self.tt), dpi=self.img_dpi)
