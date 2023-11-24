import numpy as np

np.random.seed(0)
"""
This file contains the definition of the environment
in which the agents are run.
"""


class EasyEnvironment:
    # List of the possible actions by the agents
    # possible_actions = [0,1,...,n_a]

    def __init__(self, n_a, variability):
        """Instanciate a new environement in its initial state."""
        assert variability > 0
        assert variability <= 1
        self.variability = variability
        self.ps = (np.random.rand(n_a)-0.5) * self.variability + 0.5 
        self.ps = self.ps.clip(0,1)
    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        return np.random.binomial(1, self.ps[action])

class HardEnvironment:
    # List of the possible actions by the agents
    # possible_actions = [0,1,...,n_a]

    def __init__(self, n_a, variability):
        """Instanciate a new environement in its initial state."""
        
        assert variability > 0
        assert variability <= 1
        self.variability = variability
        self.ps = np.random.rand(n_a)  # uniform in [0;1]
        self.offsets = -np.random.rand(n_a)  # uniform in [-1;0]
        self.scales = np.power(
            10, np.random.rand(n_a) * 2 - 1
        )  # log-uniform random between 0.1 and 10

    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        return (
            (np.random.binomial(1, self.ps[action]) + self.offsets[action])
            * self.variability
            * self.scales[action]
        )
