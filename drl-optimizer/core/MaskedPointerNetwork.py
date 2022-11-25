""" 
    Pointer Networks with Masking.
    Simillar to the original pointer network but we use masking to avoid pointing twice to the same input.
    
    https://arxiv.org/abs/1506.03134
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2020"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "---"
__status__ = "Production"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

from RDN import EmbeddingLSTM as EmbeddingLSTM


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, directions=2):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(directions*dim, directions*dim)
        self.project_ref = nn.Conv1d(directions*dim, directions*dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        
        v = torch.FloatTensor(2*dim) 
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)) , 1. / math.sqrt(dim))
    
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(0, 2, 1)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits


class MaskedPointerNetwork(nn.Module):
    def __init__(self,  
                in_features=1,
                hidden_size=64,
                batch_size=1,
                output_length=1,
                sos_symbol=-1,
                num_glimps=1,
                device='cpu'):

        super(MaskedPointerNetwork, self).__init__()

        # attributes
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sos_symbol = sos_symbol
        self.output_length = output_length
        self.num_glimps = num_glimps

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # encoder/decoder
        # embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
        # embedding_layer=nn.Conv1d(in_features, hidden_size, 1, 1),
        #                        embedding_type='Conv1d'     
        
        self.encoder = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
                                embedding_type='Linear',   
                                batch_size=self.batch_size,
                                hidden_size=self.hidden_size,
                                device=self.device
                                )
        # in figure 1b, we need to perfrom the decorder step by step
        # this is a curcial piece of pointer network and the main difference with seq2seq models
        # see the forward function for more details
        self.decoder_cell = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=2*hidden_size),
                                batch_size=self.batch_size,
                                hidden_size=2*self.hidden_size,
                                device=self.device,
                                lstm_cell=True # LSTMCell
                                )

        # attention calculation paramaters see first lines in equation 3 in 
        # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
        # where e_j and d_i are the encoder and decoder hidden states, respectively.
        # W_1, and W_2 are learnable weights(square matrices), we represent them here by nn.Linear layers
        # v is also a vector of learnable weights, we represent it here by nn.Paramater

        # self.W_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.W_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.v = nn.Linear(in_features=hidden_size, out_features=1)

        self.glimps = [Attention(hidden_size) for _ in range(self.num_glimps)]
        for glimp in self.glimps:
            glimp.to(self.device)
        self.pointer = Attention(hidden_size, use_tanh=False)

        self.to(self.device)

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.encoder.batch_size = batch_size
        self.decoder_cell.batch_size = batch_size

    def forward(self, input_seq, masks=None):
        """
        Calculate the attention and produce the pointers
        
        keyword argumenents:
        input_seq -- the input sequence (batch_size, sequence_size, hidden_size)
        """
        batch_size, input_seq_length, feature_size = input_seq.shape
        masks = torch.tensor(masks).to(self.device)
        masks = masks.unsqueeze(0)
        ## get the encoder output
        encoder_output, hidden, embdds = self.encoder(input_seq) # it returns output, hidden, embed
        h_n, h_c = hidden
        

        ## get the decoder output
        # 1- we start by inserting the random or zeros to the encoder_cell
        # 2- the pointer network will produce a pointer from the softmax activation
        # 3- We use that pointer to select the embedded features from the input sequence
        # 4- we feed these selected features to the encoder_cell and get a new pointer
        # 5- we iterate the above steps until we collect pointers with the same size as the input
        
        # we will use them to calculate the loss function
        pointers = torch.empty((self.batch_size, self.output_length))
        attentions = torch.empty((self.batch_size, self.output_length, input_seq_length))

        # the initial state of the decoder_cell is the last state of the encoder
        decoder_cell_hidden = (hidden[0][-1, :, :], hidden[1][-1, :, :])  # each of size(num_layers=1, batch_size, hidden_size)
        decoder_cell_hidden = (torch.reshape(hidden[0], (1, 2*self.hidden_size)), torch.reshape(hidden[1], (1, 2*self.hidden_size)))  # each of size(num_layers=1, batch_size, hidden_size)
       
        # initialize the first input to the decoder_cell, zeros, random, or using the sos_symbol
        decoder_cell_input = (torch.ones((self.batch_size, self.in_features)) * self.sos_symbol).to(self.device)
        #masks = torch.zeros((self.batch_size, input_seq_length)).to(self.device) # to avoid pointing twice to the same element
        for i in range(self.output_length):
            # 1 - calculate decoder hidden and cell states 
            
            decoder_cell_output, decoder_cell_hidden_state, _ = self.decoder_cell(decoder_cell_input, decoder_cell_hidden, clear_state=False)
            decoder_cell_hidden = (decoder_cell_output, decoder_cell_hidden_state) # because it is an LSTMCell

            # 2 - used decoder_cell_output and encoder_output to calculate the attention:
            # u^i_j = v^\top tanh(W_1 e_j + W_2 d_i), \forall j \in (1, \cdots, n)
            #u = self.v(torch.tanh(self.W_1(encoder_output) + self.W_2(decoder_cell_output).unsqueeze(1))).squeeze(2)
            # mask by a large negative number so the softmax will be close to zero
            #u -= masked * 1e6 
            
            # glimps and pointer
            q = decoder_cell_output
            for j in range(self.num_glimps):
                ref, u = self.glimps[j](q, encoder_output)
                u -= (1 - masks) * 1e6 
                q = torch.bmm(ref,F.softmax(u, dim=1).unsqueeze(-1)).squeeze(-1)
            _, u = self.pointer(q, encoder_output)
            #######################
            
            u -= (1 - masks) * 1e6 
            attentions[:, i, :] = F.softmax(u, dim=1)
            _, max_pointer = attentions[:, i, :].max(dim=1)
            
            # store the pointer
            pointers[:, i] = max_pointer
            
            # update mask
            # masked[range(self.batch_size), max_pointer] = 1

            # create a new input
            # can be refactored to a single line but this is more readable
            decoder_cell_input = decoder_cell_input.clone()
            for j in range(self.batch_size):
                decoder_cell_input[j, :] = input_seq[j, max_pointer[j], :]
        
        return attentions, pointers, _, encoder_output, h_n, embdds


