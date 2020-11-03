#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sa_env import Action
from environment import Environment
import numpy as np


class BaselinePolicy(object):
    """
    defining the baseline policy for the environment
    """

    def __init__(self, st_fac_safe, qty_fac, st_war_saf, qty_war):
        """
        :param st_fac_safe: safety stock in the factory
        :param qty_fac: qty to be reordered (from production building) for items are out of stock in factory
        :param st_war_saf: safety stock in warehouse
        :param qty_war: amount to be reordered from factory.
        """
        self.st_fac_safe = st_fac_safe
        self.qty_fac = qty_fac
        self.st_war_saf = st_war_saf
        self.qty_war = qty_war

    def choose_action(self, state):
        """
        :param state: state instance defining the current state we are at.
        :return:action performed
        """
        action = Action(state.no_of_warehouses)
        for i in range(state.no_of_warehouses):
            if state.warehouse_stock[i] < self.st_war_saf[i]:
                action.shippings[i] = self.qty_war[i]

        if state.factory_stock - np.sum(action.shippings) < self.st_fac_safe:
            action.production_level = self.qty_fac
        else:
            action.production_level = 0
        return action
