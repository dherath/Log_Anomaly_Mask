import os
import sys
import math
import random

import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import util_fileprocessor as ufp
import model_DeepLog as DeepLog

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------
#       Replay Memory
#-------------------------------------------------------------------

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    """
    Replay Memory : a cyclic buffer
    """
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        """ Save the transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """ Returns a random batch from its memory"""
        return random.sample(self.memory,batch_size)

    def __len__(self):
        """ Returns current size of memory"""
        return len(self.memory)

#--------------------------------------------------------------------
#       DQN (univariate-LSTM) Q function approximator
#--------------------------------------------------------------------

class DQN(nn.Module):

    def __init__(self,input_feature_size,hidden_size,num_layers,num_outputs):
        super(DQN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_feature_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_outputs)

    def forward(self,input):
        h0 = torch.zeros(self.num_layers,input.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,input.size(0),self.hidden_size).to(device)
        out, _ = self.lstm(input,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

#--------------------------------------------------------------------
#       LAM : Log Anomaly Mask
#---------------------------------------------------------------------

class LAM:
    """
    Illustrative example of Log Anomaly Mask used to attack a sample HDFS
    logs against whitebox DeepLog model. The perturber uses the DQN model
    defined above and uses DeepLog as the surrogate model to calculate the
    reward.
    """

    def __init__(self,n_input_features,n_hidden_size,n_layers,n_actions,window_size,state_size,model_name,EPS_START,EPS_END,EPS_DECAY,BATCH_SIZE,GAMMA):
        # initialize a policy_network and the target_network
        self.policy_net = DQN(n_input_features,n_hidden_size,n_layers,n_actions).to(device)
        self.target_net = DQN(n_input_features,n_hidden_size,n_layers,n_actions).to(device)
        
        # load policy net into target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # initialize the optimizer, memory, gloabal step and the model name
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(20000) # uses the Memory class
        self.steps_done = 0
        self.n_input_features = n_input_features
        self.n_actions = n_actions
        self.window_size = window_size
        self.state_size = state_size
        self.model_name = model_name
        self.EPS_DECAY = EPS_DECAY
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def get_eps_threshold(self):
        return self.eps_threshold
        
    def select_action(self,state,is_optimal_action):
        """ selects action used to attack using policy_net() DQN"""
        #-- if optimal action is set, then must do argmax Q
        if is_optimal_action == True:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
        #-- if not optimal action is set then can do something random as well
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                output = self.policy_net(state).detach()
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],device=device,dtype=torch.long)
    
    def optimize_DQN(self):
        """ optimizers/trains the policy_net() DQN"""
        if len(self.memory) < self.BATCH_SIZE:
            return
        # if memory is sufficient start training
        transitions = self.memory.sample(self.BATCH_SIZE)
        # convert batch-array of Transitions to Transition of batch-array
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which buffer ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool) # there was a warning here uint8
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None],dtype=torch.float).view(-1,self.state_size,self.n_input_features).to(device)

        # seperate the arguments from the memory into their own tensors
        state_batch = torch.tensor(batch.state,dtype=torch.float).view(-1,self.state_size,self.n_input_features).to(device)
        action_batch = torch.tensor(batch.action,dtype=torch.long).view(-1,1).to(device) 
        reward_batch = torch.tensor(batch.reward,dtype=torch.float).view(-1,1).to(device)
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)
        
        # Compute target action value, based on target policy (fixed network)
        next_state_values = torch.zeros(self.BATCH_SIZE,dtype=torch.float,device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # compute expected Q values
        expected_state_action_values = (next_state_values.view(-1,1) * self.GAMMA) + reward_batch
        
        # Compute the Loss (Huber Loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        return loss/self.BATCH_SIZE

    def push_to_memory(self,*args):
        """ push to replay memory"""
        self.memory.push(*args)

    def update_target_net(self):
        """updates the target network with policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def attack(self,state,next_logkey,modified_buffer,is_optimal_action):
        """ gives the state transitions from current_state -> next_state """
        dqn_state_input = torch.tensor(state,dtype=torch.float).view(-1,self.state_size,self.n_input_features).to(device)
        action = self.select_action(dqn_state_input,is_optimal_action).detach()
        next_state = state.copy()
        changed_state = state.copy()
        action_taken = action.cpu().numpy()[0,0] # actions are between 0 & (n_actions)
        modification_made = 0 # type zero, nothing
        if action_taken < self.n_actions - 1:
            # last action value if for removal/dropping logkey
            logkey = action_taken + 1
            # change the logkey is its not the same
            if state[-1] != logkey:
                modification_made = 1 # type 1 : replace to another key
                # replace logkey
                changed_state[-1] = int(logkey)
                next_state[-1] = int(logkey)
                modified_buffer[-1] = int(logkey)
            next_state.pop(0) # remove the first logkey
        else:
            modification_made = 2 # type 2 : remove the logkey
            next_state.pop()
            modified_buffer.pop()
            # changed_state == next_state
            # skips a turn here becuase a logkey is dropped
            changed_state = next_state.copy()
            changed_state.append(next_logkey)
        next_state.append(next_logkey) # append the next logkey
        modified_buffer.append(next_logkey)
        return changed_state, next_state, action_taken, modification_made, modified_buffer

    def reward_DeepLog(self,modification_flag,changed_state,DeepLog_params):
        """ new reward function"""
        # note that attack removes/modifies the final log key
        # if removal is done : changed_state == next_state
        # else changed_state != next_state 
        # because next_logkey has not been added yet
        reward = 0.0
        if modification_flag > 0:
            reward -= 0.5
        parsed_input = [int(x-1) for x in changed_state]
        model_state = DeepLog_params[0]
        num_candidates = int(DeepLog_params[1])
        Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,self.window_size,self.n_input_features,num_candidates,0)
        if Anomaly == True:
            reward = -1.0
        else:
            reward += 1.0
        return reward

    def scaled_attack(self,anomaly_buffers,DeepLog_params):
        """ Test attack on all anomaly buffers"""
        # Note: will change a state only if state is an anomaly
        advesarial_modified_buffer = [] # the advesarially modified buffers
        modifications_per_buffer = []
        for anomaly_buffer in anomaly_buffers:
            state = []
            next_state = []
            modified_buffer = []
            num_modifications_made = 0
            # append the first window_size + 1 values
            for i in range(self.state_size):
                state.append(anomaly_buffer[i])
                modified_buffer.append(anomaly_buffer[i])
            # append the rest of the logkeys based on action taken and if state is an anomaly
            episode = anomaly_buffer[self.state_size:len(anomaly_buffer)].copy() + [-1] # for final value
            for next_logkey in episode:
                if next_logkey == -1:
                    break
                parsed_input = [int(x-1) for x in state]
                model_state = DeepLog_params[0]
                num_candidates = int(DeepLog_params[1])
                Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,self.window_size,self.n_input_features,num_candidates,0)
                if Anomaly == True:
                    # is_optimal_action = True
                    changed_state, next_state, action_taken, modification_made, modified_buffer = self.attack(state,next_logkey,modified_buffer,True)
                    if modification_made > 0:
                        num_modifications_made += 1
                else:
                    # do nothing
                    next_state = state.copy()
                    next_state.pop(0)
                    next_state.append(next_logkey)
                    modified_buffer.append(next_logkey)
                # state transtion state -> next_state
                state = next_state.copy()
            advesarial_modified_buffer.append(modified_buffer)
            modifications_per_buffer.append(num_modifications_made)
        return advesarial_modified_buffer, modifications_per_buffer

    def save_policy_net(self):
        """ saves the policy_net """
        torch.save(self.policy_net.state_dict(),os.getcwd()+'/rl_model/'+self.model_name+'.pt')
        return

    def load_policy_net(self,model_name):
        """ loads a policy network already trained"""
        model_path = os.getcwd() + '/rl_model/' + model_name + '.pt'
        self.policy_net.load_state_dict(torch.load(model_path,map_location=device))
        self.policy_net.eval()
        return
