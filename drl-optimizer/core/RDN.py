#!/usr/bin/env python
""" 
    Implementation of the Recurrent Deep Network (RDN) components
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2022"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "----"
__status__ = "Dev"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(10)



class EmbeddingLSTM(nn.Module):
    """ A generic LSTM encoder """
    def __init__(self,
                 batch_size,
                 device,
                 embedding_layer, # nn.Embedding, nn.Linear, nn.Conv1d, ...
                 embedding_type='Linear', # 'Linear', 'Embedding', 'Conv1d'
                 hidden_size=64, # features
                 dropout_rate=0.0,
                 lstm_cell=False,
                 bidirectional=False,
                 ):

        super().__init__()

        # attributes
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        
  
        # layers
        self.embedding = embedding_layer
        self.embedding_type = embedding_type
        if lstm_cell:
            self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # put the model in the device
        self.to(self.device)

    
    def forward(self, x, hidden_state_and_cell=None, clear_state=True, init_random=False):
        """ The encoder which takes:

        keyward arguments:
        x -- input, expected to be a tensor being loaded in the same device as the model
        hidden_state_and_cell -- (tuple) the initial hidden state and hidden cell state
        clear_state -- If True then hidden_state_and_cell will be initialized according to init_random
        init_random -- If True then hidden_state_and_cell will be initialized with randoms, zeros otherwise
        """

        # hidden state initialization
        if clear_state:
            if init_random:
                state = torch.randn(2, self.batch_size, self.hidden_size)
            else:
                state = torch.zeros(2, self.batch_size, self.hidden_size)
            state = state.to(self.device)
            hidden_state_and_cell = (state, state)
     
        if self.embedding_type is 'Conv1d': # fix the input dim
            embd = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            embd = self.embedding(x) # first step
        
        encoder_output, hidden_state_and_cell = self.lstm(embd, hidden_state_and_cell)
        return encoder_output, hidden_state_and_cell, embd
    




    






    


