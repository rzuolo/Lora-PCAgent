""" 
    Implementaion of pointer-critic algorithm
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

from AbstractAgent import AbstractAgent


import numpy as np


class PCAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space,
                 actor_critic,
                 episode_length,
                 discount_factor=0.99,
                 entropy_factor=0.01,
                 name='PCAgent'):

        super(PCAgent, self).__init__(state_size=state_size,
                                   action_space=action_space, name=name)
        
      
        self.actor_critic = actor_critic
        self.discount_factor = discount_factor
        self.entropy_factor = entropy_factor
        self.episode_length = episode_length

        # episode rollout storage
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropy = []
        self.times = []

    def get_action(self, state, masks):
        _, _, action = self.actor_critic.stochastic_predict(state, masks)
        return action

    def get_policy_action(self, state, masks):
        # get action from the distribution
        # store the outcome into the rolout storage variables

        # do a forward pass on the ac network and collect output
        
                
        v, time, action, log_probs, entropy = self.actor_critic.collect(state, masks)
             
        #print("actions ",action, v, time )
        # store them, 
        # the reward will be added in the learn function.
        # the reason is that the reward is not avialable now
        self.values.append(v)
        self.times.append(time)
        self.log_probs.append(log_probs)
        self.entropy.append(entropy)
        
        
        return action, time
    
    def learn(self, *args):
        """ The actual algorithm of DQN goes here
        
        Keyword arguments:
        *arg -- an experience sequence sent from the episode manageer. It should be unpacked to with this order: 
            step: the step in the episode
            state: the current state 
            state_: next state
            reward: the reward 
            done: if it was a terminal state
            extras: any application dependant observations, usually it is None and is ignored
        """

        # unpack args
        steps, episode_step, _, state_, reward, _, done, masks = args 

        # we check if we should do a learning step,
        # otherwise, just store the reward and go on

        if done or episode_step == self.episode_length - 1:
            
            # get the last value
            if done: # terminal state
                last_value = 0 
            else: # predict the last value from the critic
                last_value, _ = self.actor_critic.predict(state_, masks)

            # calculate the discounted reward
            self.rewards.append(reward)
            discounted_rewards = np.zeros(len(self.values))
            for i in reversed(range(len(self.rewards))):
                last_value = self.rewards[i] + self.discount_factor * last_value
                discounted_rewards[i] = last_value
            
            # update weights by calculating the loss and performing backward
            self.actor_critic.calc_loss(discounted_r=discounted_rewards,
                                        values=self.values,
                                        times=self.times,
                                        log_probs=self.log_probs,
                                        entropy=self.entropy,
                                        entropy_factor=self.entropy_factor)
            # clean up
            self.values.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.entropy.clear()
            self.times.clear()
        else:
            self.rewards.append(reward)












        
