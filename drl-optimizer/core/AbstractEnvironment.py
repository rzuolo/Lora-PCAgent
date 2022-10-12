""" 
    An abstract envrionement class
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2022"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "---"
__status__ = "Production"

 
class AbstractEnvironement():
    def __init__(self,
                 state_size,
                 action_space):
        self.state_size = state_size
        self.action_space = action_space
    
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

