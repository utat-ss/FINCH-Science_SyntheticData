"""
Define the diffusion class here
"""

class cond_diffusion:

    def __init__(self, epsilon, scheduler):
        self.epsilon = epsilon # Take in the epsilon network
        self.scheduler = scheduler # Take in the noise scheduler

    def _gather_stats(self, t):
        """
        Gathers relevant stats from the scheduler.
        """



    def train(self, x_0, ab):
        """
        Trains the epsilon network, takes in some initial x_0 and some abundances related to it.
        """

    def sample(self, x_T, ab):
        """
        Samples from the 
        """