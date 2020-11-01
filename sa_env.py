#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class Action(object):
    """
    shipping to warehouses and changes in production level
    """

    def __init__(self, no_of_warehouses):
        self.production_level = 0  # b(0,t)
        self.shippings = np.repeat(0, no_of_warehouses)  # W(j,t)


class State(object):
    """
    defining the state of the environment
    """

    def __init__(self, no_of_warehouses, demand_history, eps, t=0):
        self.no_of_warehouses = no_of_warehouses  # W
        self.demand_history = demand_history
        self.factory_stock = 0  # l(0,t)
        self.warehouse_stock = np.repeat(0, no_of_warehouses)  # l(j,t)
        self.eps = eps
        self.t = t

    def array_state(self):
        return np.concatenate(([self.factory_stock], self.warehouse_stock, np.hstack(self.demand_history), [self.t]))

    def stock_only(self):
        return np.concatenate(([self.factory_stock], self.warehouse_stock))
