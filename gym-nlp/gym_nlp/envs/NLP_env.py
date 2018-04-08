#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate VGDL Games
"""
import os
import sys
import numpy as np


import gym
from gym import error, spaces, utils

class NLP_Env(gym.Env):
    """
    Define a VGDL environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, game, lvl):

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            state (image) :
                An image of the current frame of the game
            reward (float) :
                Total reward (Philip: Should it be incremental reward? Check Atari)
            isOver (bool) :
                whether it's time to reset the environment again.
            info (dict):
                info that can be added for debugging
        """

        #Create a function to get state, reward and isOver
        #state, reward, isOver = self.GVGAI.step(action)

        return state, reward, isOver, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        return #self.GVGAI.reset()

    def render(self, mode='human', close=False):
        #Add rendering capability
        #If we add render, add close
        return self.img
