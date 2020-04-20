import os
import sys
import math
import random
import time

import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
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

#--------------------------------------------------------
#       Parameters
#--------------------------------------------------------
model_name = 'DQN_NONBTS_nontanh_'
dataset_name = 'hdfs_DeepLog_'
debug_flag_on = False
writer_flag_on = True

if len(sys.argv) < 8:
    print('Error: argv length mismatch :',len(sys.argv))
    print('sys.argv[1] : GAMMA')
    print('sys.argv[2] : BATCH_SIZE')
    print('sys.argv[3] : EPS_DECAY')
    print('sys.argv[4] : #LSTM_layers')
    print('sys.argv[5] : #LSTM_hidden_size')
    print('sys.argv[6] : #EPOCHS (episodes)')
    print('sys.argv[7] : TARGET_UPDATE')
    sys.exit(0)

# Tunable parameters ------------------------------------

num_episodes = int(sys.argv[6]) # num iterations
warmup_data_name = 'data/warmup_data'
DeepLog_model_ensemble = DeepLog.load_ensemble(os.getcwd()+'/model/DeepLog_batch_size=4096_epoch=100_',['v1'],[1,64,2,28,9])
DeepLog_hdfs = DeepLog_model_ensemble[0]
num_candidates = 9
DeepLog_model_params = [DeepLog_hdfs,num_candidates]

training_data_name = 'data/hdfs_train' # used to create the greedy policy

#delta = 0.1 # the small negative reward for taking aciton
f_min = 1e-25 # very small value

# exploitation - eploration decay parameters
BATCH_SIZE = int(sys.argv[2]) # to train per epoch
GAMMA = float(sys.argv[1])
EPS_START = 0.9 # starting value
EPS_END = 0.05 # end value
EPS_DECAY = int(sys.argv[3]) 
TARGET_UPDATE = int(sys.argv[7]) # updates the model
TP_TEST = 50 # test for TP rate

# number of actions
num_logkeys = 28
n_actions = num_logkeys + 1 #indices = [0-27] : swap to key, 28 : remove/drop logkey

# model params
window_size = 10  # will be same as DeepLog or model used
state_size = window_size + 1 #
n_input_features = 1 # make it univariate now
n_hidden_size = int(sys.argv[5])
n_layers = int(sys.argv[4])

# update name with RL params
model_name += dataset_name+'GAMMA_'+str(GAMMA)+'_batchSz_'+str(BATCH_SIZE) + '_EPSdecay_' + str(EPS_DECAY)
# update nsme with DL params
model_name += '_layers_'+ str(n_layers) + '_hiddenSz_' + str(n_hidden_size) +'_num_epoch_'+str(num_episodes) + '_targetUpdate_' + str(TARGET_UPDATE)
#print(model_name)

writer = ''
if writer_flag_on:
    writer = SummaryWriter(logdir='rl_data/'+model_name)


#--------------------------------------------------------
#       Replay Memory
#--------------------------------------------------------

Transition = namedtuple('Transition',('state','action','next_state','reward','greedy_Q_value'))

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

#--------------------------------------------------------
#       DQN (univariate-LSTM) Q function approximator
#--------------------------------------------------------

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
        #out = torch.tanh(out)
        return out

#--------------------------------------------------------
#       RL agent: uses the DQN (keep it in same file)
#--------------------------------------------------------

class RL_agent:

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
        greedy_Q_value_batch = torch.tensor(batch.greedy_Q_value,dtype=torch.float).view(-1,1).to(device)
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)
        
        # Compute target action value, based on target policy (fixed network)
        next_state_values = torch.zeros(self.BATCH_SIZE,dtype=torch.float,device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # compute expected Q values (changed to compare between two policies {a fixed target, greedy policy})
        #expected_state_action_values = (torch.max(next_state_values.view(-1,1),greedy_Q_value_batch) * self.GAMMA) + reward_batch
        expected_state_action_values = (next_state_values.view(-1,1) * GAMMA) + reward_batch
        
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
        #if next_logkey == None or next_logkey == -1 :
        #    return
        dqn_state_input = torch.tensor(state,dtype=torch.float).view(-1,self.state_size,self.n_input_features).to(device)
        action = self.select_action(dqn_state_input,is_optimal_action).detach()
        next_state = state.copy()
        changed_state = state.copy()
        action_taken = action.cpu().numpy()[0,0] # actions are between 0 & (n_actions-1)
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

    def reward_DeepLog_type_1(self,modification_flag,changed_state,DeepLog_params):
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

#    def reward_DeepLog_type_2(self,modification_flag,next_state,delta,modified_buffer,DeepLog_params):
#        """ the first reward f() type: latent reward with small negative reward for actions """
#        # per action there are two conditions to satisfy if DeepLog is used
#        reward = 0
#        if next_state != None:
#            # modification flag = {1,2} for replace/remove
#            if modification_flag > 0:
#                reward = -1 * np.abs(delta)
#        else:
#            parsed_input = [int(x-1) for x in modified_buffer]
#            model_state = DeepLog_params[0]
#            num_candidates = int(DeepLog_params[1])
#            Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,self.window_size,self.n_input_features,num_candidates,0)
#            if Anomaly == True:
#                reward = -1
#            else:
#                reward = 1
#        return reward

    def scaled_attack(self,anomaly_buffers,DeepLog_params):
        """ Test attack on all anomaly buffers"""
        # Note: will change a state only if state is an anomaly
        advesarial_modified_buffer = [] # the advesarially modified buffers
        time_for_attack = [] # to log the speed
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
                time_1 = time.time()
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
                time_2 = time.time()
                delta_time = time_2 - time_1
                time_for_attack.append(delta_time)
            advesarial_modified_buffer.append(modified_buffer)
            modifications_per_buffer.append(num_modifications_made)
        return advesarial_modified_buffer, time_for_attack , modifications_per_buffer

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


#--------------------------------------------------------
#       The greedy removal policy
#--------------------------------------------------------

class greedy_policy:

    def __init__(self,filename,num_logkeys,f_min,window_size,n_input_features,GAMMA):
        frequency_count = np.zeros(num_logkeys)
        # populate with only the count
        with open(filename,'r') as f:
            for line in f.readlines():
                # reduce by one to get indices [0,num_logkeys-1]
                line = list(map(lambda n: n-1,map(int, line.strip().split())))
                for i in range(len(line)):
                    frequency_count[line[i]] += 1
                    #trans_matrix[line[i],line[i+1]] += 1
        # get normalized frequency
        sum_ = np.sum(frequency_count)
        frequency_count /= sum_
        self.frequency_count = frequency_count
        # figure out the logkey to be removed
        removal_logkeys = []
        for i in range(num_logkeys):
            if self.frequency_count[i] < f_min:
                removal_logkeys.append(i+1)
        self.removal_logkeys = removal_logkeys
        self.f_min = f_min
        #self.delta = delta
        self.window_size = window_size
        self.n_input_features = n_input_features
        self.GAMMA = GAMMA

    def get_removal_logkeys(self):
        return self.removal_logkeys

    def get_Qvalue_with_DeepLog(self,next_state,future_logkeys,DeepLog_params):
        #greedy_buffer = modified_buffer.copy()
        if next_state is None:
            return
        # if next_state is not none go forward
        reward_values = []
        greedy_state = next_state.copy()
        greedy_next_state = next_state.copy()
        if future_logkeys[-1] != -1:
            print('error: greedy_policy.get_Qvalue_with_DeepLog()')
            sys.exit(0)
        # use greedy policy to move forward
        for next_logkey in future_logkeys:
            reward = 0.0
            #if next_logkey in self.removal_logkeys drop that logkey
            # next_state == current_state
            if greedy_state[-1] in self.removal_logkeys:
                if next_logkey == -1:
                    break
                greedy_state[-1] = next_logkey
                reward -= 0.5
                greedy_next_state = greedy_state.copy()
            else:
                # dont do anything
                greedy_next_state = greedy_state.copy()
                greedy_next_state.pop(0)
                greedy_next_state.append(next_logkey)
            # check if action made a difference    
            parsed_input = [int(x-1) for x in greedy_state]
            model_state = DeepLog_params[0]
            num_candidates = int(DeepLog_params[1])
            Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,self.window_size,self.n_input_features,num_candidates,0)
            if Anomaly == True:
                reward = -1.0
            else:
                reward += 1.0
            reward_values.append(reward)
            # state -> next_state
            greedy_state = next_state.copy()
        # compute the Q value using all the future reward values
        greedy_action_value = 0
        reward_values.reverse()
        for r in reward_values:
            greedy_action_value = r + self.GAMMA * greedy_action_value
        return greedy_action_value
                
#    
#    def get_Qvalue_with_DeepLog(self,modified_buffer,future_logkeys,DeepLog_params):
#        #print('gQ modified buffer: ',modified_buffer)
#        #print('gQ future logkeys: ',future_logkeys)
#        greedy_buffer = modified_buffer.copy()
#        reward_values = []
#        #print('---')
#        if future_logkeys[-1] != -1:
#            print('error: greedy_policy.get_action_values() future_logkeys[-1] != -1')
#            sys.exit(0)
#        # use the greedy policy to propagate forward in the episode
#        for next_logkey in future_logkeys:
#            #print('gQ: ',next_logkey)
#            #print('-- DEBUG FROM HERE --')
#            # future logkeys will have an additional -1 added to it
#            if next_logkey != -1:
#                #print('+ next_logkey != -1')
#                if next_logkey not in self.removal_logkeys:
#                    #print('+ appended ',next_logkey)
#                    greedy_buffer.append(next_logkey)
#                    #print('+ greedy_buffer: ',greedy_buffer)
#                    reward_values.append(0.0)
#                else:
#                    reward_values.append(-1*np.abs(self.delta))
#            else:
#                parsed_input = [int(x-1) for x in greedy_buffer]
#                model_state = DeepLog_params[0]
#                num_candidates = int(DeepLog_params[1])
#                Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,self.window_size,self.n_input_features,num_candidates,0)
#                if Anomaly == True:
#                    reward_values.append(-1.0)
#                else:
#                    reward_values.append(1.0)
#                break
#        reward_values.reverse() # reverse to traverse it backwards, from final reward up till now
#        #print(reward_values)
#        #sys.exit(0)
#        action_value = 0
#        for r in reward_values:
#            action_value = r + self.GAMMA * action_value
#        return action_value

    def scaled_attack(self,anomaly_buffers):
        advesarial_buffers = []
        num_modifications = 0
        sz = 0
        for temp_buffer in anomaly_buffers:
            sz += len(temp_buffer)
            adv_buffer = []
            for x in temp_buffer:
                if x not in self.removal_logkeys:
                    adv_buffer.append(x)
                else:
                    num_modifications += 1
            advesarial_buffers.append(adv_buffer)
        avg_num_mod = num_modifications / len(anomaly_buffers)
        avg_sz = sz / len(anomaly_buffers)
        return advesarial_buffers, avg_num_mod , avg_sz

#------------------------------------------------------------------------------------------------
#       running the code
#------------------------------------------------------------------------------------------------

# the anomaly dataset used to train the model
warmup_anomaly_buffers = ufp.read_from_file(warmup_data_name)

# the reinforcement learning agent
RL = RL_agent(n_input_features,n_hidden_size,n_layers,n_actions,window_size,state_size,model_name,EPS_START,EPS_END,EPS_DECAY,BATCH_SIZE,GAMMA)

# the greedy removal policy
phi_greedy = greedy_policy(training_data_name,num_logkeys,f_min,window_size,n_input_features,GAMMA)

random.seed(100)
sum_reward = 0
TP = 0

for i_episode in range(num_episodes):
    # randomly pick a sample from the anomaly
    rand_buffer_index = random.randint(0,len(warmup_anomaly_buffers)-1)
    episode = warmup_anomaly_buffers[rand_buffer_index]
    # initialize for the state and the advesarial_modified buffer
    modified_buffer = []
    state = []
    next_state = []
    # fill in with the first few logkeys
    for i in range(state_size):
        state.append(episode[i])
        modified_buffer.append(episode[i])
    # do the attack and training for the rest of the logkeys
    future_logkeys = episode[state_size:len(episode)] + [-1]
    start_index = 0 # the position to start when creating the episode for greedy_policy implementation
    #--- debugging
    temp_batch_state = []
    temp_batch_changed_state = []
    temp_batch_next_state = []
    temp_batch_reward = []
    temp_batch_action = []
    temp_batch_greedy_Q_value = []
    temp_batch_modificaiton_type = []
    if debug_flag_on:
        print('episode: ',episode)
        print('modified buffer: ',modified_buffer)
        print('future logkeys: ',future_logkeys)
        print('---')
    #--------------
    for next_logkey in future_logkeys:
        #print(': ',next_logkey)
        if next_logkey == -1:
            changed_state, next_state, action_taken, modification_flag , modified_buffer = RL.attack(state,next_logkey,modified_buffer,False)
            next_state = None
            if changed_state[-1] == -1:
                changed_state.pop()
                modified_buffer.pop()
                changed_state = [modified_buffer[-state_size-1]] + changed_state
            reward = RL.reward_DeepLog_type_1(modification_flag,changed_state,DeepLog_model_params)
            greedy_policy_action_value = 0 # wont matter
            #----- debugging
            if debug_flag_on:
                print(': ',next_logkey)
                print('state: ',state)
                print('next state: ',next_state)
                print('---')
                temp_batch_reward.append(reward)
                temp_batch_greedy_Q_value.append(greedy_policy_action_value)
                temp_batch_state.append(state)
                temp_batch_changed_state.append(changed_state)
                temp_batch_next_state.append(next_state)
                temp_batch_modificaiton_type.append(modification_flag)
                temp_batch_action.append(action_taken) # cannot have a -1 indexed action
            #---------------
            RL.push_to_memory(state,action_taken,next_state,reward,greedy_policy_action_value)
            sum_reward += reward
            break
        else:
            changed_state, next_state, action_taken, modification_flag, modified_buffer = RL.attack(state,next_logkey,modified_buffer,False)
            rest_of_episode = future_logkeys[start_index+1:len(future_logkeys)] # the start_index logkey is already added into the modified buffer
            greedy_policy_action_value = phi_greedy.get_Qvalue_with_DeepLog(next_state,rest_of_episode,DeepLog_model_params)
            reward = RL.reward_DeepLog_type_1(modification_flag,changed_state,DeepLog_model_params)
            #---- debugging
            #print('future logkeys: ',future_logkeys[start_index:len(future_logkeys)])
            #print('rest of episode: ',rest_of_episode)
            if debug_flag_on:
                print(': ',next_logkey)
                print('state: ',state)
                print('next_state: ',next_state)
                print('--')
                temp_batch_reward.append(reward)
                temp_batch_state.append(state)
                temp_batch_changed_state.append(changed_state)
                temp_batch_next_state.append(next_state)
                temp_batch_modificaiton_type.append(modification_flag)
                temp_batch_action.append(action_taken)
                temp_batch_greedy_Q_value.append(greedy_policy_action_value)
            #-------------
            RL.push_to_memory(state,action_taken,next_state,reward,greedy_policy_action_value)
            state = next_state.copy()
            sum_reward += reward
            start_index += 1
        loss = RL.optimize_DQN()
        eps_threshold = RL.get_eps_threshold()
        # add to tensorboardx
        if writer_flag_on:
            if loss is not None:
                writer.add_scalar('eps_threshold',eps_threshold,i_episode+1)
                writer.add_scalar('train_loss',loss.cpu(),i_episode+1)
                writer.add_scalar('sum_reward',sum_reward,i_episode+1)
    # update the target_net with policy_net in RL agent
    if i_episode % TARGET_UPDATE == 0:
        RL.update_target_net()
    # save the policy network every 1000 epochs
    if i_episode % 1000 == 0:
        RL.save_policy_net()
    # calculate the True Positive rate for warmup_data
    if i_episode % TP_TEST == 0:
        TP = 0
        #random_buffer_sample = random.sample(warmup_anomaly_buffers,50)
        advesarial_buffers, _ , _  = RL.scaled_attack(warmup_anomaly_buffers,DeepLog_model_params)
        for advesarial_buffer in advesarial_buffers:
            parsed_input = [int(x-1) for x in advesarial_buffer]
            model_state = DeepLog_model_params[0]
            num_candidates = int(DeepLog_model_params[1])
            Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,window_size,n_input_features,num_candidates,0)
            if Anomaly == True:
                TP += 1
        TP = TP / len(warmup_anomaly_buffers)
        if writer_flag_on:
            writer.add_scalar('TP_rate (warmup_data)',TP,i_episode+1)
        else:
            print('i: ',i_episode,' TP: ',TP)
    #---- debugging
    if debug_flag_on:
        # debugging print statements
        print('sz state:',len(temp_batch_state))
        print('sz next_state:',len(temp_batch_next_state))
        print('sz changed_state: ',len(temp_batch_changed_state))
        print('sz greedy_Q:',len(temp_batch_greedy_Q_value))
        print('sz actions: ',len(temp_batch_action))
        print('sz mod.made: ',len(temp_batch_modificaiton_type))
        print('sz rewards:',len(temp_batch_reward))
        # debugging print statements
        for i in range(len(temp_batch_reward)):
            s1 = temp_batch_state[i]
            s_ = temp_batch_changed_state[i]
            s2 = temp_batch_next_state[i]
            r = temp_batch_reward[i]
            a = temp_batch_action[i]
            gQ = temp_batch_greedy_Q_value[i]
            print(i,' ',s1,' ',a,' ',s_,' -> ',s2,' ',r,' ',gQ)
        print('---------------------------------------------------------------------')
    #--------------
    #sys.exit(0)

# save the model
RL.save_policy_net()
if writer_flag_on:
    writer.close()

