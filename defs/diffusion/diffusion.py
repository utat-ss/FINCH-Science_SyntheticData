"""
Define the diffusion class here
"""

import numpy as np

class cond_diffusion:

    def __init__(self, epsilon, scheduler):
        self.epsilon = epsilon # Take in the epsilon network
        self.scheduler = scheduler # Take in the noise scheduler

    def _gather_stats(self, t):

        """
        Gathers relevant stats from the scheduler. Not that needed
        """

        self.beta_t = self.scheduler.beta_t(t)
        self.beta_tilda_t = self.scheduler.beta_tilda_t(t)

    def _scheduled_call(self, x_0):

        """
        Randomly samples some noised data given some initial x_0, with given scheduler.
        """

        return self.scheduler.add_noise(x_0, round(np.random.uniform(low=0, high= self.scheduler.steps)))

    def train_sample(self, x_0, ab):

        """
        Trains the epsilon network for a single sample, takes in some initial x_0 and some abundances related to it.
        """

    def sample(self, x_T, ab):

        """
        Samples from the 
        """