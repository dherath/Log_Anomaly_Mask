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

import LAM as attack_model

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------
#       Parameters
#--------------------------------------------------------
model_name = 'exp1_'
dataset_name = 'hdfs_DeepLog_'
debug_flag_on = False
writer_flag_on = False

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
all_data_name = 'data/hdfs_abnormal_test_complete_data'

DeepLog_model_ensemble = DeepLog.load_ensemble(os.getcwd()+'/model/DeepLog_batch_size=4096_epoch=100_',['v1'],[1,64,2,28,9])
DeepLog_hdfs = DeepLog_model_ensemble[0]
num_candidates = 9
DeepLog_model_params = [DeepLog_hdfs,num_candidates]

training_data_name = 'data/hdfs_train' # used to create the greedy policy

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

#------------------------------------------------------------------------------------------------
#       running the code: test phase (DeepLog = HDFS)
#------------------------------------------------------------------------------------------------

# the anomaly dataset used to train the model
all_anomaly_buffers = ufp.read_from_file(all_data_name)

# the reinforcement learning agent
LAM = attack_model.LAM(n_input_features,n_hidden_size,n_layers,n_actions,window_size,state_size,model_name,EPS_START,EPS_END,EPS_DECAY,BATCH_SIZE,GAMMA)
LAM.load_policy_net(model_name)

TP = 0
i_episode = 0

advesarial_buffers, modifications_per_buffer = LAM.scaled_attack(all_anomaly_buffers,DeepLog_model_params)

avg_modifications_per_buffer = np.average(modifications_per_buffer)
std_modifications_per_buffer = np.std(modifications_per_buffer)
modifications_per_buffer = []

for advesarial_buffer in advesarial_buffers:
    parsed_input = [int(x-1) for x in advesarial_buffer]
    model_state = DeepLog_model_params[0]
    num_candidates = int(DeepLog_model_params[1])
    Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state,parsed_input,window_size,n_input_features,num_candidates,0)
    if Anomaly == True:
        TP += 1
TP = TP / len(all_anomaly_buffers)
if writer_flag_on:
    writer.add_scalar('TP_rate (all_data)',TP,i_episode+1)
else:
    string_text = ''
    string_text += model_name + '\n'
    string_text += '+ True Positive rate (after attack): ' + str(TP) + '\n'
    string_text += '+ #modifications (per session): ' + str(avg_modifications_per_buffer) + ' +/- ' + str(std_modifications_per_buffer) + '\n'
    print(string_text)

ufp.write_list_of_lists('results/adversarially_modified_logs.txt', advesarial_buffers) # stores the changed adversarial buffer
if writer_flag_on:
    writer.close()
