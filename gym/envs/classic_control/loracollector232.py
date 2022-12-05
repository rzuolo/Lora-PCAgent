#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:43:48 2022

@author: zuolo
"""


import numpy as np 
from gym import Env,spaces


class LoraCollector232(Env):
  
    def __init__(self):
                
        ### Observation is composed of 5 entries: 
        ### (1) The column identifying the cluster position of the gateway. Which is the same as action selected. 
        ### This is 0 for all lines, except the line where the gateway is positioned that is assigned with 1
        ### (2) The column indicating the time cost for moving from the previous position to the current position. 
        ### This is given by time steps.
        ### (3) This column indicates the ammount of data collected in the current position.
        ### This amount is in bytes and is reffered as data in the buffer.
        ### (4) This column denotes the time to live of the data carried in the buffer.
        ### After this time is elpased, if not dumped, it expires. The time if given in timesteps and it is trasnversall to all lines.
        ### (5) This column denotes the amount of data remaining to be collected in the current cluster/position.
        ### This number is given in bytes, and in case of depot/upload, it denotes the amount uploaded in that point
        self.observation_space = spaces.Discrete(5)  # Postion time buffer timer reamining_bytes
        
        ## Each action signifies a position in the graph of clusters. Regardless they being depot or cluster for collection.
        self.action_space = spaces.Discrete(8) #0 1 2 3 ... 8     
        self.action_space.n = 8
        
        
        # Maximum flags for common timer, episode, airbourn cargo, etc
        self.max_time = 862 #game time insteps
        #self.max_time = 868
        self.max_tick = 540 #ttl in timesteps
        self.max_cargo = 3000 #maximum allowed buffer (large number because is not in use currently)
        self.period =  200  #app period in seconds (this is the frequency that every node will be sending packets).
        self.time_elapsed = 0 #time elapsed (in timesteps) of the game
        #####################################################
        #####################################################
        ### Number of vertices in the graph
        ### 1 means it is a cluster for collecting data
        ### 0 means is a depot to store/upload data
        self.vertices = np.loadtxt("vertices_ext.txt").reshape(8)
        #self.vertices = [1,1,1,1,0,0,0]
        
        #Msg sending frequency bootup time
        #After every start which is in the startup_ext, the nodes will be sending packets for every self.period time cycle
        #these values are only to indicate when the nodes start sending their first packets (in seconds)
        self.node_freq = np.loadtxt("startup_ext.txt").reshape(8,20)
        
        #self.node_freq = [ [20,15,40,0,0,0,0,0,0],
        #                   [0,0,0,20,40,0,0,0,0],
        #                   [0,0,0,0,0,80,90,15,0],
        #                   [0,0,0,0,0,0,0,0,10]]
   

        # The total number of nodes can be extracted from the frequencies array
        self.nodes = len(self.node_freq[0]) 

        ####################################################
        ####################################################
        #### Timesptes table, considering step of 5 seconds
        ####################################################
        ####################################################
        #590 = 9.3 minutes  118 timesteps
        #1500 = 25 minutes   300 timesteps
        #600  = 10 minutes   120 timesteps
        #299  = 4.9 minutes  59.8 timesteps
        #2100 = 35 minutes   420 timesteps
        #328  = 5.4 minutes  65.6 timesteps
        #2621 = 43.6 minutes 524.2 timesteps
        #780 = 13 minutes    156 timesteps
        #1200 = 20 minutes   240 timesteps
        
        # The amount of bytes sent per message for each node
        # The number of indices here needs to match the length of node_freq
        #self.node_payload = [10,100,10,100,10,100,10,100,10]
        self.node_payload = [10]*self.nodes
        
        ###############################################################################################
        # Number of packets to be collected for each node
        # The index represents the node #, whilst its content represnts the amount of messages(packets)
        ###############################################################################################
        ###############################################################################################
        self.node_target = np.loadtxt("targets_ext.txt").reshape(self.nodes)
        #self.node_target = [1,2,3,2,4,2,5,3,3]
        #e.g. in the above, the node #3 is node_target[2], which demands 3 packets to be collected.

        ##################################################################
        ############### Matrix for movement duration #####################
        ############### Graph vertices travel times  #####################
        ############### Times are given in timesteps #####################
        ##################################################################
        self.move_duration = np.loadtxt("triptimes.txt").reshape(8,8)
        starting_cluster = [-1]*8
        self.move_duration = np.vstack((self.move_duration,starting_cluster))
        #self.move_duration = [ 
        #                [0,288,504,576,115,518,432],
        #                [288,0,216,115,115,173,56],
        #                [504,216,0,144,230,115,84],
        #                [576,115,144,0,461,29,230],
        #                [115,115,230,461,0,20,20],
        #                [518,173,115,29,20,0,20],
        #                [432,56,86,230,20,20,0],
        #                [-1,-1,-1,-1,-1,-1,-1] 
        #               ]
        
        ### used to determine if a node has been visited at game over
        ### only for debugging
        self.checked = [0]*self.nodes        
    
        ################################################################### 
        ### set maximum values to be used during input normalization
        ###
        self.max_costtime = 4
        self.max_buffer   = 500
        self.max_earliest = self.max_tick + 1
        
        
        self.visit_time = 100
   



    def reset(self):
        
        self.actions_available = 8
        self.gateways_available = 1

        self.state = np.zeros(shape=(8,5))
        #reset masks
        self.masks = np.ones_like(self.state[:, 0])

        
        #########################################################################
        #### Create all counters and flags according with the scale of gateways 
        #########################################################################
        self.buffer = 0
        self.game_over = [0]
        ## this indicates the time the data has being carried in the buffer (timesteps) for TTL calculation
        ## it starts the game with the maxtime of the the game
        self.earliest = self.max_tick+1
        self.total_reward = 0 
        self.visited = [0]*self.nodes        
        self.time_elapsed = 0
       
        
        ### Calculate the maximum expected for all clusters
        self.max_volume = 0
        for idx in range(self.actions_available):
            self.max_volume = self.max_volume + self.total_expected_volume(idx)
        
        ### Initialize the remaining (expected bytes)
        ### to be colleceted from every cluster
        for cluster in range(self.actions_available):
            self.state[cluster][4] = self.normalize_input(self.total_expected_volume(cluster),500)            
         
        return self.state, self.masks
    

    #######################################################
    ######### normalize features between 0 and 1
    #######################################################
    def normalize_input(self,input,max):
        if input/max >= 0 and input/max <= 1:
            return input/max
        else:
            print("There was an error normalizing the input data input ",input," max ",max, " return ", input/max)
            return 0
        
    
    #######################################################
    ######### update the earliest timer seen for a gateway
    #######################################################
    def tick_update(self,time):
        if time < self.earliest: 
            self.earliest = time
            
                
    #######################################################
    ######### check if the time limit has been breached
    #######################################################
    def tick_check(self,time):
        if time-self.earliest >= self.max_tick:
            return 0
        else:
            return 1

    ########################################################
    ######## Check if the node is upstreaming data
    ######## at specific time and position
    ########################################################
    def sender_active(self,time,node,vertex,wait):
        #print("LOG4 Node is ",node, " and time is ",time, " ", (self.node_freq[vertex][node])+self.period, " and pos is ", vertex)
        #print ("LOG4 This is the freq ",(self.node_freq[vertex][node]*60)/5)
        #wait=100
        # there must be a time conversion here because 1 timestep = 5 seconds
#        if ( (self.node_freq[vertex][node] != 0)  and ((time*5) > (self.node_freq[vertex][node])+self.period)  and (   ((time*5) % (self.node_freq[vertex][node])) <= 5) ):
        if ( (self.node_freq[vertex][node] != 0)  and ((time*5) > ((self.node_freq[vertex][node])+self.period))):
            self.checked[node] = 1 
            self.tick_update(time)
            points=0
            if  ((time*5)-self.node_freq[vertex][node]) % (self.period) == 0:
                points=points+1    
            #if  ((((time*5+wait)-self.node_freq[vertex][node]) % (self.period) == 0) and (wait !=0)):
            #    points=points+1    
            
            a = int((time*5-(self.node_freq[vertex][node]))/(self.period))
            b = int((((time+wait)*5)-self.node_freq[vertex][node])/(self.period))
            points = points + int(b-a)
            
            
        #and (   ((time*5) % (self.period) <= 5) )):
        #if ( time % node_freq[node-1]) <= (time-5):
            if (points > 0) and (self.visited[node]  < self.node_target[node]) :
                if (self.visited[node] + points) <= self.node_target[node]:
                    self.buffer = self.buffer + (self.node_payload[node] * points)
                    self.visited[node] = points + self.visited[node]
                    return self.node_payload[node]*points
                else:
                    points = self.node_target[node] - self.visited[node]
                    self.buffer = self.buffer + (self.node_payload[node] * points)
                    self.visited[node] = self.node_target[node]
                    return self.node_payload[node]*points
            else:
              return 0
        else:
              return 0
               
            
                    #print("LOG5 The sender interval is active ",node," Gateway ", 0 ," payload ",self.node_payload[node]," time ",time, " position ",self.state[0]," visitado ", self.visited[node], " target ", self.node_target[node])
                    #print("LOG5 Node is ",node, " and time is ",time, " ", (self.node_freq[vertex][node])+self.period, " and pos is ", vertex)
                    #print ("LOG5 This is the points ", points)
                    #print ("LOG5 This is the freq ",(self.node_freq[vertex][node]*60)/5)
                    #print ("LOG5 This is start time ",(self.node_freq[vertex][node]))
                    #print ("LOG5 This is the real time ", time*5)
                    #print ("LOG5 This is the real endtime ", ((wait+time)*5))
                    #print ("LOG5 This is the real return ", ((10)*points))
                    # print ("LOG5 Target, Achieved", self.node_target[node], self.visited[node])
    #                return self.node_payload[node]
            
            
              
        

    ########################################################
    ######## Check if the node is at an upstreaming depot
    ######## reward if arriving on time. (penalize otherwise?)
    ########################################################
    def dump_buffer(self,time):
        
#        print("LOG Dumping buffer ",self.buffer)
        if self.buffer == 0:
            return 0     
        else:
            #return 10
            if self.tick_check(time) == 1 :
                #print("LOG4 The gateway is dumping            Gateway ", 0 ," payload ",self.buffer      ," time ",time, " position ",self.state[0])
                self.earliest = self.max_tick+1
                return self.buffer
            else:
                self.earliest = self.max_tick+1
                #print("LOG4 The buffer EXPIRED", self.buffer, " position ",self.state[0], " Gateway ", 0 )
                return 0     
                #return -1*(self.buffer[gw])
    


    ##############################################################################
    ######## Calculate the total bytes that a particular cluster is supposed 
    ######## to issue throughout the game. It is expected that 
    ######## the agent collects this amount of data for that respective cluster
    ##############################################################################
    def total_expected_volume(self,cluster_idx):
        
        expected_volume = 0
        #for cluster in range(self.action_space.n):
        for node in range(self.nodes):
            if self.node_freq[cluster_idx][node] != 0:
                expected_volume = expected_volume + (10*self.node_target[node])
        #print("This is the expected volume for cluster ",cluster_idx," volume ",expected_volume)                    
        return expected_volume
                
    
    ##############################################################################
    #receive the mask tensor and return the action number
    #it checks the mask and return the selected one
    ##############################################################################
    def decouple_mask(self,turn):
        action_masked = (turn == 0).nonzero()
        #print ("action masked ",action_masked[0])
        action_masked = int(action_masked[0])
        return action_masked
    
    ######################################################
    ######################################################
    #move the gateways, update their states
    #apply rewards and return the resulting environment
    ######################################################
    ######################################################
    def step(self, vaction):
                
        #going = self.action_breakdown(action,self.gateways_available,self.actions_available)
        
        action            = vaction[0]
        self.visit_time   = vaction[1]

        
        self.masks = np.ones_like(self.state[:, 0]) 
        self.masks[action] = 0
        
        
        #print(" action time",action,self.visit_time)        

            
        #going,mascara = action, self.masks 
        #time_coming = 0
        pos_coming  = 0
        time_coming = 0
        time_going  = 0 
        pos_going   = 0
        
        
        gain = 0
        #matrix update
                
        #print(" uncoupled masks ", self.masks  )
        #print(" Decoupling masks ", self.decouple_mask(turn)  )
        
        #Ontain the action selected accoding to the mask
        turn = self.decouple_mask(self.masks)
        
        
        ## confirm the values of the state it is moving from (time_coming,pos_coming)
        for x in range(8): 
            if self.state[x][0] == 1:
                pos_coming  = x
                #time_coming = self.state[x][1] ## remove this if time difference is based on unit step costs
        #version based on cost time         
        time_coming = self.time_elapsed  
        
        
        pos_going,time_going = self.move(time_coming,pos_coming,turn,self.visit_time) #moving to new position and time
        #print("LOG7 timecoming ",timecoming[turn]," poscoming ", coming[turn])
        #print("LOG7 timegoing ",time_going," posgoing ", pos_going)

        if np.count_nonzero(self.game_over) == self.gateways_available:
            over = 1
            #return self.state,gain,over,self.masks
        
        
        #update the new time current cost + time elpased so far
        #self.time_elapsed = self.time_elapsed + time_going

                  
        time_going_normalized = self.normalize_input(time_going,self.max_time)
        #print("Time going", time_going, " time going normalized ", time_going_normalized )
        ##updating the state with the new positions and times
        ## here we run a loop to update every single line of the matrix
        for x in range(8): 
            self.state[x][0] = 0
            self.state[x][1] = time_going_normalized
            #self.state[x][1] = time_going 
        self.state[pos_going][0] = 1 

       
        
        ###############################################
        ###############################################
        ### Reading all new times from the states,
        ### running through the existing gateways.
        ### Check their position and time
        ### in relation to all nodes (second loop).
        ### Verify if they will be fecth or dump. 
        ### Apply rewards and scores accoringly.
        ###############################################
        ###############################################
        
        for node in range(self.nodes): 
         
            ## If the position the gateways is moving to is
            ## is a cluster, then verify if there are packets to collect
            if self.vertices[pos_going] == 1:
                if time_going <= self.max_time:
                     gain  = gain + self.sender_active(time_going,node,pos_going,self.visit_time)
                
                     
            ## If the position the gateways is edge
            ## confirm if the data upload matches the requirements
            else:
             if self.vertices[pos_going] == 0:
               if  time_going <= self.max_time:
                     gain = gain + self.dump_buffer(time_going)
                     self.buffer = 0 
               
               
               
        ### update the remaining bytes to be collected in each cluster
        ### or update the amount of uploaded bytes on each depot
        if self.vertices[pos_going] == 1:
            #print("subtraindo ganhos", self.state[pos_going][4]-self.normalize_input(gain,self.max_volume))
            #print("subtraindo ganhos", 500*self.state[pos_going][4],"Ganho de ",gain)
            self.state[pos_going][4] = self.state[pos_going][4] - self.normalize_input(gain,self.max_volume)         
        else:
            self.state[pos_going][4] = self.state[pos_going][4] + self.normalize_input(gain,self.max_volume)
            
        #update the buffer number with the amount of data corresponding that cluster (line number in the state matrix) 
        self.state[pos_going][2] = self.state[pos_going][2] + self.normalize_input(gain,self.max_buffer)
        
        ##convert it back into the state format and store ttl to it
        for x in range(8): 
            self.state[x][3] = self.normalize_input(self.earliest,self.max_earliest)
        #self.state[len(self.state)-1] = turn
        
        ### dummy execution to prove convergence
        #if pos_going == 5:
        #    gain = -20
        #else:
        #    gain = 10
       
        #apply a penalty for every non profitable step
        #if gain == 0:
        #    gain = -1
            
        ##update the reward
        #if gain > 0:
            #print("LOG4 This is the supposed reward ",reward)
        self.total_reward = self.total_reward + gain
        
        ##confirm the end of the game
#        if np.count_nonzero(self.game_over) == self.gateways_available:
#            over = 1
#        else:
#            over = 0
#        
        if (self.game_over[0] == 1):
            over = 1
        else:
            over = 0    
        
        #terunt the steps output
        #print ("Action taken ",pos_going, self.masks)
        return self.state,gain,over, self.masks 
    
      
    
    ## When a step is taken the positions and time need to be updated
    def move(self,time_coming,coming,going,wait):
              
        if coming == going:
#            print("Do nothing, stays at same position")
            new_state_idx = coming
        else:
            new_state_idx = going 
        
        ## this is valid if new_time is based on the elapsed time
        
        if self.vertices[going] == 1:
            new_time_idx = self.time_elapsed + self.trip_time(coming,going) + 1 + wait #using a wait time of 20    
        else:
            new_time_idx = self.time_elapsed + self.trip_time(coming,going) + 1 #using a wait time of 20    
        ## in this new version, the time is simply the cost to moving between vertices
        #new_time_idx = self.trip_time(coming,going) + 1    
        
        if new_time_idx >= self.max_time:
             new_time_idx = self.max_time
             self.game_over[0] = 1 #this code was changed for multiple gws to a single gateway. Now this tensor has only one element wihch is [0]       
        
        self.time_elapsed = new_time_idx
        
        return new_state_idx,new_time_idx
    
    
    def trip_time(self,prevstate,state):
        #print("Moving between states : -- from :", int(prevstate)," : -- to : " ,int(state) )    
        return (self.move_duration[int(prevstate)][int(state)]) 
