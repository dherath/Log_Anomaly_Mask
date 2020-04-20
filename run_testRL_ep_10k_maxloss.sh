#!/bin/bash

# order python attack_RL_DQN_maxloss.py GAMMA BATCH_SIZE EPS_DECAY #LSTM_layers #LSTM_hiddenSZ #EPOCHS

# 1. Keep eps decay , #epochs constant

# --- BATCH_SIZE = 256

python test_RL_DQN_maxloss.py 0.85 256 2000 2 64 10000 
python test_RL_DQN_maxloss.py 0.90 256 2000 2 64 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 2 64 10000 

python test_RL_DQN_maxloss.py 0.85 256 2000 2 128 10000   
python test_RL_DQN_maxloss.py 0.90 256 2000 2 128 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 2 128 10000 

python test_RL_DQN_maxloss.py 0.85 256 2000 2 256 10000 
python test_RL_DQN_maxloss.py 0.90 256 2000 2 256 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 2 256 10000 

python test_RL_DQN_maxloss.py 0.85 256 2000 4 64 10000 
python test_RL_DQN_maxloss.py 0.90 256 2000 4 64 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 4 64 10000 

python test_RL_DQN_maxloss.py 0.85 256 2000 4 128 10000 
python test_RL_DQN_maxloss.py 0.90 256 2000 4 128 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 4 128 10000 

python test_RL_DQN_maxloss.py 0.85 256 2000 4 256 10000 
python test_RL_DQN_maxloss.py 0.90 256 2000 4 256 10000 
python test_RL_DQN_maxloss.py 0.95 256 2000 4 256 10000 
#
## --- increased BATCH_SIZE to 512

python test_RL_DQN_maxloss.py 0.85 512 2000 2 64 10000 
python test_RL_DQN_maxloss.py 0.90 512 2000 2 64 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 2 64 10000 

python test_RL_DQN_maxloss.py 0.85 512 2000 2 128 10000  
python test_RL_DQN_maxloss.py 0.90 512 2000 2 128 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 2 128 10000 

python test_RL_DQN_maxloss.py 0.85 512 2000 2 256 10000 
python test_RL_DQN_maxloss.py 0.90 512 2000 2 256 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 2 256 10000 

python test_RL_DQN_maxloss.py 0.85 512 2000 4 64 10000 
python test_RL_DQN_maxloss.py 0.90 512 2000 4 64 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 4 64 10000 

python test_RL_DQN_maxloss.py 0.85 512 2000 4 128 10000 
python test_RL_DQN_maxloss.py 0.90 512 2000 4 128 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 4 128 10000 

python test_RL_DQN_maxloss.py 0.85 512 2000 4 256 10000 
python test_RL_DQN_maxloss.py 0.90 512 2000 4 256 10000 
python test_RL_DQN_maxloss.py 0.95 512 2000 4 256 10000 
