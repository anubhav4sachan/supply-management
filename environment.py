#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import collections

from sa_env import State, Action


class Environment(object):
    """
    defining the reward distribution system
    and the next state based on the appropriate action and
    current state.
    """

    def __init__(self, eps=30, W=4, max_demand=5, r_var=2, c=35, p=98):
        """
            Initializing the initial conditions of the system.
        """
        self.eps = eps  # Epochs (transitions)
        self.no_of_warehouses = W
        self.d_max = max_demand  # maximum demand at warehouse w -> d(j, t)
        self.d_var = r_var  # maximum random variation in units in d(j,t) for different warehouses
        self.unit_cost = c  # c_0 -> production cost
        self.unit_price = p  # p->product price for retailers

        ''' storage capacity at the factory, storage capacity at each warehouse, rupees per unit
            first index because of factory warehouse followed by distribution warehouses'''
        self.storage_capacities = np.fromfunction(lambda j: 10 * (j + 1), (self.no_of_warehouses + 1,), dtype=int)

        ''' storage capacity at the factory, storage capacity at each warehouse, rupees per unit
            first index because of factory warehouse followed by distribution warehouses'''
        self.storage_costs = np.fromfunction(lambda j: 2 * (j + 1), (self.no_of_warehouses + 1,),
                                             dtype=int)

        # transportation costs for each warehouse, dollars per unit
        self.transporation_costs = np.fromfunction(lambda j: 5 * (j + 1), (self.no_of_warehouses,),
                                                   dtype=int)
        self.penalty_unit_cost = self.unit_price

        self.reset()

    def reset(self, demand_history_len=4):
        """
        :param demand_history_len: defines number of past records to be tracked
        """
        self.t = 0
        self.demand_history = collections.deque(maxlen=demand_history_len)

        for _ in range(demand_history_len):
            self.demand_history.append(np.zeros(self.no_of_warehouses))

    def demand(self, t, j):
        """
        :param t: time at which demand is needed
        :param j: warehouse for which demand is needed
        :return: returns demand at time t for warehouse j
        """
        return np.round(
            self.d_max / 2 + self.d_max / 2 * np.sin(2 * np.pi * (t + 2 * j) / self.eps * 2) + np.random.randint(0,
                                                                                                                 self.d_var))

    def initial_state(self):
        """
        :return: returns initial state
        """
        return State(no_of_warehouses=self.no_of_warehouses, demand_history=list(self.demand_history), eps=self.eps)

    def step(self, state, action):
        """
        :param state: state instance
        :param action: action instance
        :return: next_state,rewards,completed episodes or not
        """
        demands = np.fromfunction(lambda j: self.demand(self.t, j + 1), (self.no_of_warehouses,), dtype=int)

        # calculating the reward (profit)
        total_revenue = self.unit_price * np.sum(demands)
        production_cost = self.unit_cost * action.production_level
        total_storage_cost = np.dot(self.storage_costs,
                                    np.maximum(state.stock_only(), np.zeros(self.no_of_warehouses + 1)))

        transportation_cost = np.dot(self.transporation_costs, action.shippings)
        penalty_cost = -1 * self.penalty_unit_cost * (
                np.sum(np.minimum(state.warehouse_stock, np.zeros(self.no_of_warehouses))) + min(
            state.factory_stock, 0))

        rewards = total_revenue - production_cost - total_storage_cost - penalty_cost - transportation_cost

        # Calculating next state
        next_state = State(no_of_warehouses=self.no_of_warehouses, demand_history=list(self.demand_history),
                           eps=self.eps,
                           t=self.t)
        next_state.factory_stock = min(state.factory_stock + action.production_level - np.sum(action.shippings),
                                       self.storage_capacities[0])

        for i in range(self.no_of_warehouses):
            next_state.warehouse_stock[i] = min(
                state.warehouse_stock[i] + action.shippings[i] - demands[i],
                self.storage_capacities[i + 1])

        self.t += 1
        self.demand_history.append(demands)
        return next_state, rewards, self.t == self.eps - 1
