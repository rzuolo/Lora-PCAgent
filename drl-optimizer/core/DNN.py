""" 
    Typical Deep Neural Network Wrappers for torch.nn
"""
__author__ = "AL-Tam, Faroq"
__copyright__ = "Copyright 2022"
__credits__ = ["AL-Tam, Faroq"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "AL-Tam, Faroq"
__email__ = "---"
__status__ = "Production"


# torch stuff
import torch   
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# numpy
import numpy as np

# local files
from RDN import EmbeddingLSTM as EmbeddingLSTM
from MaskedPointerNetwork import MaskedPointerNetwork as MaskedPointerNetwork, Attention



class Critic(nn.Module):
    def __init__(self,
                in_features,
                hidden_size,
                batch_size=1,
                num_process_blocks=10,
                device='cpu'):

        super(Critic, self).__init__()
        # attributes
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_process_blocks = num_process_blocks

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device


        # encoder/decoder
        # embedding_layer=nn.Conv1d(in_features, hidden_size, 1, 1),
        # embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
        self.encoder = EmbeddingLSTM(embedding_layer=nn.Linear(in_features=in_features, out_features=hidden_size),
                                embedding_type='Linear',   
                                batch_size=self.batch_size,
                                hidden_size=self.hidden_size,
                                device=self.device
                                )

        # encoder is two hidden linear layers as in https://arxiv.org/pdf/1611.09940.pdf
        self.decoder = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(2*hidden_size, 1))


        
        self.timedecoder = nn.Sequential(nn.Linear(8*hidden_size, 1))
                                         
                

        
        # attention params
        # self.W_1 = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        # self.W_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.v = nn.Linear(in_features=hidden_size, out_features=1)
        self.process_block =Attention(hidden_size, use_tanh=False)
        # put into device
        self.to(self.device)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def forward(self, state, encoder_output, h_n, embdds):
        
        
        v2 = self.timedecoder(torch.reshape(embdds, (1,(8*self.hidden_size))))
        
        # call encoder, it will take care of state initialization
        encoder_output, (h_n, c_n), embdds = self.encoder(state) # it returns output, hidden, embed

        #print( "hn size ",h_n.size())
        #print( "encoder_output inside loop ",encoder_output, encoder_output.size())
        #print( "embedds ",embdds.size())
        #print( "cn ",c_n.size())
       
        
        #print(" encoder_output ",encoder_output[0,:,0])
        # process block loop
        #q = h_n.squeeze(0) # hidden state
        q = torch.reshape(h_n, (1, 2*self.hidden_size)) # hidden state
        
        #q2 = torch.reshape(h_n, (1, 2*self.hidden_size)) # hidden state
        
        #v2 = self.timedecoder(q)
        
        #print("Sigmoid 1",float(v2))
        #print("Sigmoid 2",float(F.sigmoid(v2)))
        #print("Tan ", float(F.tanh(v2)))
        
        #print(self.sigmoid(float(v2)))
        #print(self.tanh(float(v2)))
        #print(" Timedecoder ", v2)
        
        v2 = self.tanh(float(v2))
        
        
        for _ in range(self.num_process_blocks):
            ref, u = self.process_block(q, encoder_output)
            q = torch.bmm(ref,F.softmax(u, dim=1).unsqueeze(-1)).squeeze(-1)

        # decode query
        v = self.decoder(q) # decoder_output is the value function of the critic
 
        
        return v, v2
        

       

class PointerCriticArch(nn.Module):
    """ 
    DNN architecture of the pointer-critic network.
    This class fuses the pointer and critic architecture and produces required outputs for the loss functions
    used by the REINFORCE algorithm.
    """
    def __init__(self,
                in_features=1,
                hidden_size=64,
                batch_size=1,
                output_length=1,
                device='cpu',
                lr=1e-4):
        super(PointerCriticArch, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_length = output_length
        self.lr = lr

        # pointer
        self.pointer = MaskedPointerNetwork(in_features=self.in_features,
                                            hidden_size=self.hidden_size, 
                                            batch_size=self.batch_size,
                                            output_length=self.output_length,
                                            device=device)
        # critic
        self.critic = Critic(in_features=self.in_features,
                            hidden_size=self.hidden_size,
                            batch_size=self.batch_size,
                            device=device
                             )

        # device
        if device is not 'cpu':
            if torch.cuda.is_available():
                device = 'cuda:0'
        self.device = device

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)
    
    def forward(self, state, masks):
        
        state = torch.tensor(state).to(self.device)
        state = state.float().unsqueeze(0) #.unsqueeze(-1)
        
        probs, act, _, encoder_output, h_n, embdds = self.pointer(state, masks)
        
        #print("Encoder ",encoder_output, encoder_output.size())
        #print("Embdds ",embdds, embdds.size())
        #something = torch.reshape(encoder_output, (1, 16*self.hidden_size))
        #print("Encoder ",something, something.size())
        
        #print("Masks ",masks)
        
              
        probs = probs.squeeze(0).squeeze(-1)
        #v, time = self.critic(state, encoder_output, h_n, embdds).squeeze(-1)
        
        v, time = self.critic(state, encoder_output, h_n, embdds)
        
        #time = 100
        #v = 0
        return v, probs, act, time
    


##################################### warpers #########################
# NN class for Actor-critic agents
class ACDNN:
    def __init__(self,
                in_features=1,
                hidden_size=64,
                batch_size=1,
                output_length=1,
                device='cpu',
                lr=1e-4):
        
        self.model = PointerCriticArch(in_features=in_features,
                                    hidden_size=hidden_size,
                                    batch_size=batch_size,
                                    output_length=output_length,
                                    device=device,
                                    lr=lr)

   
        
    def predict(self, source, masks):
        v, probs, _, _ = self.model(source, masks)
        return v.detach().cpu().item(), probs.detach().cpu().numpy()
    
    def stochastic_predict(self, source, masks):
        v, probs, action, time = self.model(source, masks)
        dist = torch.distributions.Categorical(probs)

        #action = dist.sample(). this is used during the training
        action = dist.sample()
        #action = torch.argmax(probs)
        return v.detach().cpu().item(), probs.detach().cpu().numpy(), action.detach().cpu().item()

    def collect(self, source, masks):
        
        #print("source-masks ", source, masks )
        v, probs, action, time = self.model(source, masks)
              
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        
        #print("SAMPLE1 ",dist.sample() )
        #print("SAMPLE2 ",dist.entropy().mean() )
        #print("SAMPLE2 ",dist.sample() )
        #print("SAMPLE2 ",dist.entropy().mean() )
                
        entropy = dist.entropy().mean()
        #entropy = torch.zeros(1)
        return v, time, action.cpu().detach().item(), log_probs, entropy
    
    
    def calc_loss(self,
                    discounted_r,
                    values,
                    log_probs,
                    entropy,
                    entropy_factor=0.01):
        """ Calculate the loss function and do the backward step

        keyword arguments:
        discounted_r -- the estimated Q in the Advantage equation: A_t(s, a) = r_{t+1} + gamma v_{t+1}(s) - v_t(s)
        values -- the esitmated values produced by the ciritic model
        log_probs -- the log of the distribution of the actions produced by the actor model
        entropy -- the entropy term which is used to encourage exploration. It is calcualted from probs
        entropy_factor -- is the contribution of the entropy term in the loss. Higher value means higher exploration.

        """

        discounted_r = torch.from_numpy(discounted_r).to(self.model.device)
        values = torch.stack(values).to(self.model.device)
        log_probs = torch.stack(log_probs).to(self.model.device)
        entropy = torch.stack(entropy).sum().to(self.model.device)
        # normalize discounted_r
        # critic loss
        adv = discounted_r.detach() - values
        critic_loss = 0.5 * adv.pow(2).mean()
        #critic_loss = F.smooth_l1_loss(values.double(), discounted_r.detach())

        # actor loss
        actor_loss = -(log_probs * adv.detach()).mean()

        loss = actor_loss - entropy_factor * entropy + critic_loss
        
        # reset grads
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        del entropy
        del log_probs
        del values
        del discounted_r

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
