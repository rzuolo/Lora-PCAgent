#%% 
from .core.DNN import ACDNN
from .core.PCAgent import PCAgent
 # import your env

import gym


import numpy as np

title = 'lora using pointer critic'
###### main components of the scenario go here ######


 # env = 
                                    
state_size = env.observation_space.shape[0]
action_space = [i for i in range(env.action_space.n)]

state_type = np.int16
action_type = np.int16

####### training options to be used by the training manager #######
num_episodes = 2000
episode_length = 100 * n_prbs
log_file = 'scenario_name_log_file.txt'

# neural nets
device = 'gpu'
lr = 0.00001

actor_critic = ACDNN(
            in_features=4,
            hidden_size=1024,
            lr=lr,
            device=device)

# agent
discount_factor = 0.99
entropy_factor = 0.001

agent = PCAgent(state_size=state_size, 
                 action_space=action_space,
                 actor_critic=actor_critic,
                 discount_factor=discount_factor,
                 entropy_factor=entropy_factor,
                 episode_length=episode_length 
                 )

model_path = 'pc_mac_.pt'