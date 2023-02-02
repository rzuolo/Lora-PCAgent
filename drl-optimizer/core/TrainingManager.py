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

from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import trange
import torch

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

    """ Receives the tensor output from the critci and convertes it into 
        different timesteps durations: 25, 50, 75 or 100
    """
    def timestep_converter(self,timetensor):
       

        timestep_index = int(torch.argmax(timetensor))
        #timestep_float = float(timetensor)
       
        #print(timestep_float)
        #timestep_real = int(timestep_float/0.25)
        #print(timestep_real)
       
        timestep_length = 25 + (timestep_index*25)
        #timestep_length = (timestep_real*25)
        return timestep_length

    def run(self, worldrecord, verbose=False, plot=False, save_to_file=True, parallel=False):
        """ Run the RL scenario using the settings of the TrainingManager
        
        
        Keyword arguments:
        verbose -- if True, the manager will print the total reward for each epsiode and some other statics
        plot -- if True, the manager will plot online learning curve
        parallel -- if True the manager will parallel - NOT SUPPORTED YET.
        """
        #world_record = 0
        # do some assertions first
        assert self.agent is not None, "Agent can not be None"
        assert self.env is not None, "Environment object can not be None"
        assert self.average_reward_steps > 1, "Reward must be averaged on more than 1 episode"
        
        # validate the agent is ready for training
        #assert self.agent.validate(), "Agent validation failed"

        #load the model 
        #self.agent.actor_critic.load_model("./modelsaved.pt")

        # do the magic here, i.e., the main training loop
        last_rewards = deque(maxlen=self.average_reward_steps)
        all_rewards = []
        all_average_reward = []
        total_steps = 0
        episode_repetition = 0 
        previous_episode_reward = 0 
        with open(self.log_file,mode='w') as log:
            for i in range(self.num_episodes): 
                # 1 and 2- reset the environement and get initial state
                state, masks = self.env.reset()
                # 3 get first action
                action, timetensor = self.agent.get_policy_action(state, masks)
                
                ########################################################
                ### Adpatation to define a timestep of 20 or 100 units
                #if time <= 0:
                #    time = 20 
                #else:
                #    time = 100
                time = self.timestep_converter(timetensor)
                #if time < 20:
                #    time = 20
                vaction = [action,time]
                ########################################################
                ########################################################
                
                # 4 - iterate over the episode step unitl the agent moves to a terminal state or 
                # the episode ends 
                step = 1
                epsiode_done = False
                episode_reward = 0
                episode_reward_repetition = 0
                actions_list = []
                time_list = []
                extra_signals_list =[]
                while not epsiode_done and step < self.episode_length:
                    # record actions
                    actions_list.append(action)
                    time_list.append(time)
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
                    action, timetensor = self.agent.get_policy_action(state, extra_signals)
                    ########################################################
                    ### Adpatation to define a timestep of 20 or 100 units
                    #if time <= 0:
                    #    time = 20 
                    #else:
                    #    time = 100
                    
                    time = self.timestep_converter(timetensor)
                    #if time < 20:
                    #    time = 20
                    vaction = [action,time]
                    ########################################################
                    ########################################################
                
                    
                    #print ("Action-Time ",action, time)
                    
                    step += 1
                    total_steps += 1
                if verbose:
                    #print(state[ :,4])
                    
                    #### 
                    ####Use this if you want to see how many packets are 
                    ####remaining to be collected at the end of the episode
                    ####
                    #target=self.env.node_target.tolist()
                    #result = []
                    #for i, j in zip(target,self.env.visited):
                    #    result.append(int(i - j))
                    #print(result)
                    
                    print('Episode:{}\treward:{}\tsteps:{}'.format(i, episode_reward, step))
                    
                    
                    self.agent.actor_critic.rep = 0
                    
                    #### Some jury-rigged code to improve the model saving 
                    #### when performing well
              
                    if (i%500 == 0):
                        time=datetime.now()
                        format_time="%Y-%m-%d %H:%M:%S.%f"
                        now=datetime.strptime(str(time),format_time)
                        model_file = 'saved_model_'+str(episode_reward)+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'.pt'
                        print("Saving model ")
                        self.agent.actor_critic.save_model(model_file)

                
                    if previous_episode_reward == episode_reward:
                        episode_reward_repetition = episode_reward_repetition + 1
                    else:
                        episode_reward_repetition = 0

                    previous_episode_reward = episode_reward

                    if episode_reward_repetition > 5 and episode_reward > world_record:
                        print("Saving above the record ")
                        self.agent.actor_critic.save_model("./modelsaved.pt")


                    #if episode_reward > world_record and episode_reward > 2640:
                    #    world_record = episode_reward
                    #    print("Saving new record ")
                    #    self.agent.actor_critic.save_model("./modelsaved.pt")
                
                all_rewards.append(episode_reward)
                last_rewards.append(episode_reward)
                average_reward = sum(last_rewards)/self.average_reward_steps
                all_average_reward.append(average_reward)
                log.write(str(i) +" "+ str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\tActions list:" + str(actions_list) + "\n")
                log.write(str(i) +" "+ str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\tTime list:" + str(time_list) + "\n")
                log.flush()
        
        self.agent.actor_critic.save_model("./modelsaved.pt")
        return all_rewards, all_average_reward



    def test(self, worldrecord, verbose=False, plot=False, save_to_file=True, parallel=False):
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
       
        #print("loading the model")
        #print("Model's state_dict:")
        #for param_tensor in self.agent.actor_critic.model.state_dict():
        #    print(param_tensor, "\t", self.agent.actor_critic.model.state_dict()[param_tensor].size())
        self.agent.actor_critic.load_model("./modelsaved.pt")
        #self.agent.actor_critic.model.load_state_dict(torch.load('./modelsaved'))
        self.agent.actor_critic.model.eval()
        #print("model loaded")
        
        #for param_tensor in self.agent.actor_critic.model.state_dict():
        #    print(param_tensor, "\t", self.agent.actor_critic.model.state_dict()[param_tensor].size())
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
                action, timetensor = self.agent.get_policy_action(state, masks)
                #action, time = self.agent.get_policy_action(state, masks)
                time = self.timestep_converter(timetensor)
                
                
                vaction = [action, time]
                # 4 - iterate over the episode step unitl the agent moves to a terminal state or 
                # the episode ends 
                step = 1
                epsiode_done = False
                episode_reward = 0
                actions_list = []
                time_list = []
                extra_signals_list =[]
                while not epsiode_done and step < self.episode_length:
                    # record actions
                    actions_list.append(action)
                    time_list.append(time)
                    # call step function in the environement
                    state_, reward, done, extra_signals = self.env.step(vaction)
                    #masks, jfi, thu = extra_signals
                    #extra_signals = (jfi, thu)
                    episode_reward += reward
                    if done == 1:
                        epsiode_done = True
                    extra_signals_list.append(extra_signals)
                    # next state-action pair
                    state = state_
                    action, timetensor = self.agent.get_action(state, masks)
                    time = self.timestep_converter(timetensor)
                    vaction = [action, time]
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
                log.write(str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\tTime list:" + str(time_list) + "\n")
                log.write(str(extra_signals_list) + '\n')
                #log.write(str(step)+ "\t" + str(episode_reward)+ "\t" + str(average_reward) + "\n")
                log.flush()

        return all_rewards, all_average_reward, all_extra_signals

