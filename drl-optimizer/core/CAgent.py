""" 
    Implementaion of Conventional Agents
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2022"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "---"
__status__ = "Production"

from .core.AbstractAgent import AbstractAgent
from math import floor

import numpy as np

# random agent
class RandomAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space,
                 actor_critic,
                 env:None):

        super(RandomAgent, self).__init__(state_size=state_size,
                                    action_space=action_space)

        self.env = env   
        self.name ='random'                
    def get_action(self, state, masks):
        action = np.random.choice([k for k in self.action_space if masks[k]!=0])
        if action is None:
            action = np.random.choice([k for k in self.action_space])

        self.last_action = action
        return action

    def get_policy_action(self, state, masks):
       return self.get_action(state, masks)
    
    def learn(self, *args):
        pass
    def reset(self):
        pass
