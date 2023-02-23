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
import math
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


        
        #self.timedecoder = nn.Sequential(nn.Linear(8*hidden_size, 1))
                                         
                

        
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
        
        
        #v2 = self.timedecoder(torch.reshape(embdds, (1,(8*self.hidden_size))))
        
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
        
        #v2 = self.tanh(float(v2))
        
        
        for _ in range(self.num_process_blocks):
            ref, u = self.process_block(q, encoder_output)
            q = torch.bmm(ref,F.softmax(u, dim=1).unsqueeze(-1)).squeeze(-1)

        # decode query
        v = self.decoder(q) # decoder_output is the value function of the critic

        #print("critic ",float(v))
        return v
        

       

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
        
        # time
        self.time = Time(in_features=self.in_features,
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
        
        v    = self.critic(state, encoder_output, h_n, embdds)
        time, time_probs = self.time(state, encoder_output, h_n, embdds)
        
        #time = 100
        #v = 0
        return v, probs, act, time, time_probs
    


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
        v, probs, _, time, time_probs = self.model(source, masks)
        #### add time influence onto v
        #print(f'v before:{v} time:{time}')
        #v = (v/((torch.argmax(time)/4)+0.25))
        #print(f'v after:{v2} time:{time}')
        return v.detach().cpu().item(), probs.detach().cpu().numpy()
        #return v.detach().cpu().item(), time.detach().cpu().item(), probs.detach().cpu().numpy(), time_probs.detach().cpu().numpy()
    
    def stochastic_predict(self, source, masks):
        v, probs, action, time, time_probs = self.model(source, masks)
        dist = torch.distributions.Categorical(probs)
        dist_time = torch.distributions.Categorical(time_probs)

        #action = dist.sample(). this is used during the training
        action = dist.sample()
        time = dist_time.sample ()
        #time = dist.sample()
        #action = torch.argmax(probs)
        return v.detach().cpu().item(), probs.detach().cpu().numpy(), action.detach().cpu().item(), time, time_probs.detach().cpu().numpy()

    def collect(self, source, masks):
        
        #print("source-masks ", source, masks )
        v, probs, action, time, time_probs = self.model(source, masks)
              
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        
        dist_times = torch.distributions.Categorical(time_probs)
        time = dist_times.sample()
        log_time_probs = dist_times.log_prob(time)
       
        ### leave it or sum up/avg *if needed
        entropy = dist.entropy().mean()
        #entropy = torch.zeros(1)
        return v, time, action.cpu().detach().item(), log_probs, log_time_probs, entropy
    
    
    def calc_loss(self,
                    discounted_r,
                    values,
                    times,
                    log_probs,
                    log_time_probs,
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
        times = torch.stack(times).to(self.model.device)
        log_probs = torch.stack(log_probs).to(self.model.device)
        log_time_probs = torch.stack(log_time_probs).to(self.model.device)
        entropy = torch.stack(entropy).sum().to(self.model.device)
        # normalize discounted_r
        # critic loss

        #print(f'log_time_probs:{log_time_probs} log_probs:{log_probs}')
        #print(f'values_size:{times.size()} values:{values}')
        #times = times.squeeze(1)
        #a,b = torch.max(times,dim=2)
        #print(f'times_size:{times.size()} arg_times:{((torch.argmax(times,dim=2))/4)+0.25}')
        #selected_idx=((torch.argmax(times,dim=2))+1)
        
        #print(f'times:{times}')
        #a = torch.std(times,dim=2).unsqueeze(-1)
        #times = a.squeeze(1)
        #print(f'times22:{times}')
        #a, b = torch.max(times,dim=2)
        #times = a.unsqueeze(-1)
        #times = a.unsqueeze(dim=-1)
        #print(f'values_size:{times.size()} values:{values}')
        #print(f'values:{values.size()}')
        #print(f'discounted_size:{discounted_r.detach().size()} rewards:{discounted_r.detach()}')
        #print(f'argmax:{((torch.argmax(times,dim=1).unsqueeze(dim=-1)))}')
        #adv = (discounted_r.detach() - values ) * (25+(torch.argmax(times,dim=2)*25) )
        
        #adv = (discounted_r.detach() - values )
        ####adv = (discounted_r.detach() - values + times)
        #adv = (discounted_r.detach() - values) + (times/(selected_idx/4))
        #print(f'selected:{selected_idx}')
        #print(f'argmax:{selected_idx.size()} another:{times.size()} ')
        #print(values) 
        #adv = (discounted_r.detach() - (values / (selected_idx/4)) )
        #print(f'argmax:{selected_idx} discounted:{times/(selected_idx/4)}')
        #adv = (discounted_r.detach() - values + (times/(selected_idx/4)) )
        adv = ( discounted_r.detach() - values )
        #adv = (discounted_r.detach() - values) / (times/(selected_idx/4))
        #adv = (discounted_r.detach() - values) + times 
        #adv = (discounted_r.detach())/(selected_idx/4) 
        #adv = adv - values
        #print("adv ",adv)

        #times = times.squeeze(1)
        #print(f'times.shape:{times.shape} times.std:{times.std()} times:{times}')
        #maxtimes = torch.max(times,1).values.unsqueeze(1)

        #print("maxtimes ",maxtimes)
        #adv2 = times.std() #working somehow well
        #adv2 = times.std() * adv.detach()
        
        #adv2 =(maxtimes - times.mean() ) * discounted_r.detach() ## this is doing well
        #adv2 = discounted_r.detach() - torch.max(times,1).values.unsqueeze(1)
        
        critic_loss = 0.5 * adv.pow(2).mean()
        #critic_loss = adv.mean()
        
        #critic_loss = F.smooth_l1_loss(values.double(), discounted_r.detach())

        # actor loss
        actor_loss = -( (log_probs + log_time_probs) * adv.detach()).mean()

        #time_super_loss = entropy.mean()*times.mean() 
        #print("super_time_loss ",time_super_loss)
        #print(" Time super loss ", time_super_loss.mean())
        # time loss
        #time_loss = ((times.max()-times.mean())*discounted_r.detach().mean())
        #time_loss = ((selected_idx/4))
        #print("####################################################################")
        
        #### working with no fundamental rationale
        ####
        #delta = times-times.mean() 
        #delta = delta.pow(2).mean()
        #time_loss = delta.mean()*adv.detach().mean()
        ######
        ######

        #delta = times.mean() 
        #time_loss = 0.5 * adv2.pow(2).mean()
        #time_loss =  adv2.mean()
        #time_loss = times.mean()
        #print(times,times.mean())
        #delta = times * discounted_r.detach().mean()
        

        #time_loss = delta.mean()
        #print(float(time_loss))
        #print(critic_loss)
        #time_loss = delta
        #time_loss = delta.mean()*critic_loss
        #### 
        ## 
        #time_loss = a*discounted_r.detach().mean()

        #print(time_loss)
        #print(a)
        #print(discounted_r.detach().size())
        #print(adv.size())
        #print("####################################################################")
        #time_loss = (times.mean()*0.0001*discounted_r.detach().mean())
         
        #time_loss = ((times.max()-times.mean())*adv.detach().mean())
        #print("time_loss ",times.mean()*discounted_r.mean())

        #time_loss = time_super_loss.mean()*discounted_r.mean()
        #time_loss = (times.mean()*adv.detach().mean())
        #time_loss =  times.mean()*adv.mean() 
        #time_loss = time_loss.detach()
        #time_loss = -(times * adv.detach()).mean()
        #print(" Advantage ",adv)
        #print(" reward ", discounted_r.detach().mean())
        #print(" adv ", adv.detach().mean())
        #print(" timeloss ", time_loss)
        #print(" critic_loss ", critic_loss)

        #loss = times.mean()
        loss = actor_loss - entropy_factor * entropy + critic_loss 
        #loss = actor_loss - entropy_factor * entropy + (critic_loss * time_loss)
        #loss = actor_loss - entropy_factor * entropy + critic_loss + time_loss*0.00000001 
        #loss = actor_loss - entropy_factor * entropy + critic_loss + time_loss*1000000
        #loss = time_loss 
        #print(" detached ",discounted_r.detach())
        #print(" values ",values)
        
        # reset grads
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        del entropy
        del log_probs
        del log_time_probs
        del values
        del times
        del discounted_r

    def save_model(self, path):
        #torch.save(self.model.state_dict(), path)
        torch.save(self.model, path)

    def load_model(self, path):
        #self.model.load_state_dict(torch.load(path))
        self.model = torch.load(path)
        self.model.eval()


class Time(nn.Module):
    def __init__(self,
                in_features,
                hidden_size,
                batch_size=1,
                num_process_blocks=10,
                device='cpu'):

        super(Time, self).__init__()
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
        #self.decoder = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size),
        #                            nn.ReLU(),
        #                            nn.Linear(2*hidden_size, 4))
        
        self.decoder = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size),
                                    nn.ReLU(),
                                    #nn.Linear(2*hidden_size, 2*hidden_size),
                                    #nn.ReLU(),
                                    #nn.Linear(4*hidden_size, 4*hidden_size),
                                    #nn.ReLU(),
                                    nn.Linear(2*hidden_size, 4),
                                    nn.Softmax())
                                    
                                    #nn.ReLU(),
                                    #nn.Tanh())
                                    #nn.Sigmoid())


        
        #self.timedecoder = nn.Sequential(nn.Linear(8*hidden_size, 1))
        
        
              

        # attention params
        # self.W_1 = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        # self.W_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.v = nn.Linear(in_features=hidden_size, out_features=1)
        self.process_block =Attention(hidden_size, use_tanh=False)
        # put into device
        self.to(self.device)
        
        


    #def sigmoid(self, z):
    #    return 1 / (1 + np.exp(-z))

    #def tanh(self, z):
    #    return np.tanh(z)
    
    #def tanh(self, z):
    #    return np.tanh(z)

    def forward(self, state, encoder_output, h_n, embdds):
        
        
        #v2 = self.timedecoder(torch.reshape(embdds, (1,(8*self.hidden_size))))
        
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
        
        #v2 = self.tanh(float(v2))
        
        
        for _ in range(self.num_process_blocks):
            ref, u = self.process_block(q, encoder_output)
            q = torch.bmm(ref,F.softmax(u, dim=1).unsqueeze(-1)).squeeze(-1)

        # decode query
        time_probs = self.decoder(q) # decoder_output is the value function of the critic
        time = int(time_probs.argmax())

        #print(" V em quatro ",v.detach())
        #print(" V em index ", torch.argmax(v))
        #print(" V em softmax ", F.softmax(v,dim=1).detach()) 
        #v = int(torch.argmax(v))
        #v = torch.sigmoid(v)
        #v = math.log(v)
        #v = torch.tanh(v.detach().cpu())
        #print(" v ",torch.tanh(torch.max(v)))
        #print("Time ",v)
        #print("Time mean",float(v.mean()))
        return time, time_probs


