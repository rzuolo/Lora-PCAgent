#%% 
import sys,os

directory = os.getcwd()
parentdirectory = os.path.abspath(os.path.join(directory, os.pardir))

#sys.path.insert(0,'/home/ceotuser/source/Lora-PCAgent/drl-optimizer/core')
sys.path.insert(0,directory+'/core')
sys.path.insert(0,parentdirectory)
#sys.path.insert(0,'/home/ceotuser/source/Lora-PCAgent')
#sys.path.remove('/home/ceotuser/anaconda3/lib/python3.9/site-packages')

from DNN import ACDNN
from PCAgent import PCAgent
#import yourenv
import gym
import numpy as np


#n_prbs = 20
title = 'lora using pointer critic'
###### main components of the scenario go here ######


env = gym.make('LoraCollector-v232') 
                                    
state_size = env.observation_space.shape
action_space = [i for i in range(env.action_space.n)]
state_type = np.int16
action_type = np.int16


####### training options to be used by the training manager #######
num_episodes = 1000
episode_length = 100
log_file = 'scenario_name_log_file.txt'

# neural nets
device = 'cpu'
lr = 0.001

actor_critic = ACDNN(
            in_features=5,
            #hidden_size=2048,
            #hidden_size=1024,
            hidden_size=256,
            #hidden_size=512,
            lr=lr,
            device=device)

# agent
discount_factor = 0.99
entropy_factor = 0.00001
#entropy_factor = 10000000000

agent = PCAgent(state_size=state_size, 
                 action_space= action_space,
                 actor_critic=actor_critic,
                 discount_factor=discount_factor,
                 entropy_factor=entropy_factor,
                 episode_length=episode_length 
                 )

model_path = 'pc_mac_.pt'
