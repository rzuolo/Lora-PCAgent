""" 
    Training Manager. It takes an agent, environement and run the scenario for a set of episodes.
    It show online learning  curve
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2022"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "---"
__status__ = "Dev"

from AbstractAgent import AbstractAgent
from AbstractEnvironment import AbstractEnvironement

from collections import deque # to mantian the last k rewards

import matplotlib.pyplot as plt
from tqdm import trange


class TrainingManager:
    def __init__(self,
                 num_episodes,
                 episode_length,
                 agent:AbstractAgent,
                 env:AbstractEnvironement,
                 average_reward_steps=5,
                 log_file='training_log.txt'):

        """TrainingManager initializer.

        Keyword arguments:
        num_episodes -- number of episodes
        episode_length -- length of an episode
        agent -- a RL agent, see AbstractAgent
        env  -- an environement object, see AbstractEnvironment 
        average_reward_steps -- number of last episodes where the average reward is calculated from
        device -- where will the neural networks be trained ("cpu", "gpu"). If multiple gpus exist the manager will choose the first avialable.
        """

        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.agent = agent
        self.env = env
        self.average_reward_steps = average_reward_steps
        self.log_file = log_file



    def run(self, verbose=False, plot=False, save_to_file=True, parallel=False):
        """ Run the RL scenario using the settings of the TrainingManager
        
        
        Keyword arguments:
        verbose -- if True, the manager will print the total reward for each epsiode and some other statics
        plot -- if True, the manager will plot online learning curve
        parallel -- if True the manager will parallel - NOT SUPPORTED YET.
        """
        
        # do some assertions first
        assert self.agent is not None, "Agent can not be None"
        assert self.env is not None, "Environment object can not be None"
        assert self.average_reward_steps > 1, "Reward must be averaged on more than 1 episode"
        
        # validate the agent is ready for training
        #assert self.agent.validate(), "Agent validation failed"

        # do the magic here, i.e., the main training loop
        last_rewards = deque(maxlen=self.average_reward_steps)
        all_rewards = []
        all_average_reward = []
        total_steps = 0
        with open(self.log_file,mode='w') as log:
            for i in range(self.num_episodes): 
                # 1 and 2- reset the environement and get initial state
                state, masks = self.env.reset()
                # 3 get first action
                action, time = self.agent.get_policy_action(state, masks)
                
                ########################################################
                ### Adpatation to define a timestep of 20 or 100 units
                if time <= 0:
                    time = 20 
                else:
                    time = 100
                vaction = [action,time]
                ########################################################
                ########################################################
                
                # 4 - iterate over the episode step unitl the agent moves to a terminal state or 
                # the episode ends 
                step = 1
                epsiode_done = False
                episode_reward = 0
                actions_list = []
                extra_signals_list =[]
                while not epsiode_done and step < self.episode_length:
                    # record actions
                    actions_list.append(action)
                    # call step function in the environement
                    state_, reward, done, extra_signals = self.env.step(vaction)
                    
                    
                    #print("That is one state \n", state)
                    #print("This is one action-reward \n", action, reward)
                    #print(" this is done ", done, " episode ", step)
                    
                    episode_reward += reward
                    if done == 1:
                        epsiode_done = True
                    extra_signals_list.append(extra_signals)
                    # Call learn function in the agent. To learn for the last experience
                    self.agent.learn(total_steps, step, state, state_, reward, action, done, extra_signals)
                    # next state-action pair
                    state = state_
                    action, time = self.agent.get_policy_action(state, extra_signals)
                    ########################################################
                    ### Adpatation to define a timestep of 20 or 100 units
                    if time <= 0:
                        time = 20 
                    else:
                        time = 100
                    vaction = [action,time]
                    ########################################################
                    ########################################################
                
                    
                    #print ("Action-Time ",action, time)
                    
                    step += 1
                    total_steps += 1
                if verbose:
                    print('Episode:{}\treward:{}\tsteps:{}'.format(i, episode_reward, step))
                    self.agent.actor_critic.rep = 0
                all_rewards.append(episode_reward)
                last_rewards.append(episode_reward)
                average_reward = sum(last_rewards)/self.average_reward_steps
                all_average_reward.append(average_reward)
                log.write(str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\tActions list:" + str(actions_list) + "\n")
                log.flush()
        
        return all_rewards, all_average_reward



    def test(self, verbose=False, plot=False, save_to_file=True, parallel=False):
        """ Run the RL scenario using the settings of the TrainingManager
        Keyword arguments:
        verbose -- if True, the manager will print the total reward for each epsiode and some other statics
        plot -- if True, the manager will plot online learning curve
        parallel -- if True the manager will parallel - NOT SUPPORTED YET.
        """
        
        # do some assertions first
        assert self.agent is not None, "Agent can not be None"
        assert self.env is not None, "Environment object can not be None"
        assert self.average_reward_steps > 1, "Reward must be averaged on more than 1 episode"
        
        # validate the agent is ready for training
        #assert self.agent.validate(), "Agent validation failed"

        # do the magic here, i.e., the main training loop
        last_rewards = deque(maxlen=self.average_reward_steps)
        all_rewards = []
        all_average_reward = []
        all_extra_signals = []
        total_steps = 0
        #t = trange(self.num_episodes, desc='Testing the agent', leave=True)
        with open(self.log_file, mode='w') as log:
            for i in range(self.num_episodes):
                 
                # 1 and 2- reset the environement and get initial state
                state, masks = self.env.reset()
                # 3 get first action
                action, time = self.agent.get_policy_action(state, masks)
                vaction = [action, time]
                # 4 - iterate over the episode step unitl the agent moves to a terminal state or 
                # the episode ends 
                step = 1
                epsiode_done = False
                episode_reward = 0
                actions_list = []
                extra_signals_list =[]
                while not epsiode_done and step < self.episode_length:
                    # record actions
                    actions_list.append(action)
                    # call step function in the environement
                    state_, reward, done, extra_signals = self.env.step(vaction)
                    masks, jfi, thu = extra_signals
                    extra_signals = (jfi, thu)
                    episode_reward += reward
                    if done == 1:
                        epsiode_done = True
                    extra_signals_list.append(extra_signals)
                    # next state-action pair
                    state = state_
                    action = self.agent.get_action(state, masks)
                    step += 1
                    total_steps += 1
                if verbose:
                    print('Episode:{}\treward:{}\tsteps:{}'.format(i, episode_reward, step))
                all_rewards.append(episode_reward)
                last_rewards.append(episode_reward)
                average_reward = sum(last_rewards)/self.average_reward_steps
                all_average_reward.append(average_reward)
                all_extra_signals.append([sum(x)/step for x in zip(*extra_signals_list)])
                log.write(str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\tActions list:" + str(actions_list) + "\n")
                log.write(str(extra_signals_list) + '\n')
                #log.write(str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\n")
                log.flush()

        return all_rewards, all_average_reward, all_extra_signals
