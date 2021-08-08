import sys
import os
import random

from tensorboardX import SummaryWriter

import util_fileprocessor as ufp
import model_DeepLog as DeepLog

import LAM as attack_model

# --------------------------------------------------------
#       Parameters
# --------------------------------------------------------

model_name = 'exp1_'
dataset_name = 'hdfs_DeepLog_'
debug_flag_on = False
writer_flag_on = False

if len(sys.argv) < 8:
    print('Error: argv length mismatch :', len(sys.argv))
    print('sys.argv[1] : GAMMA')
    print('sys.argv[2] : BATCH_SIZE')
    print('sys.argv[3] : EPS_DECAY')
    print('sys.argv[4] : #LSTM_layers')
    print('sys.argv[5] : #LSTM_hidden_size')
    print('sys.argv[6] : #EPOCHS (episodes)')
    print('sys.argv[7] : TARGET_UPDATE')
    sys.exit(0)

# Tunable parameters ------------------------------------

num_episodes = int(sys.argv[6])  # num iterations
warmup_data_name = 'data/warmup_data'
all_data_name = 'data/hdfs_abnormal_test_complete_data'

DeepLog_model_ensemble = DeepLog.load_ensemble(os.getcwd() + '/model/DeepLog_batch_size=4096_epoch=100_', ['v1'], [1, 64, 2, 28, 9])
DeepLog_hdfs = DeepLog_model_ensemble[0]
num_candidates = 9
DeepLog_model_params = [DeepLog_hdfs, num_candidates]

training_data_name = 'data/hdfs_train'  # used to create the greedy policy

# exploitation - eploration decay parameters
BATCH_SIZE = int(sys.argv[2])  # to train per epoch
GAMMA = float(sys.argv[1])
EPS_START = 0.9  # starting value
EPS_END = 0.05  # end value
EPS_DECAY = int(sys.argv[3])
TARGET_UPDATE = int(sys.argv[7])  # updates the model
TP_TEST = 50  # test for TP rate

# number of actions
num_logkeys = 28
n_actions = num_logkeys + 1  # indices = [0-27] : swap to key, 28 : remove/drop logkey

# model params
window_size = 10  # will be same as DeepLog or model used
state_size = window_size + 1
n_input_features = 1  # make it univariate now
n_hidden_size = int(sys.argv[5])
n_layers = int(sys.argv[4])

# update name with RL params
model_name += dataset_name + 'GAMMA_' + str(GAMMA) + '_batchSz_' + str(BATCH_SIZE) + '_EPSdecay_' + str(EPS_DECAY)
# update nsme with DL params
model_name += '_layers_' + str(n_layers) + '_hiddenSz_' + str(n_hidden_size) + '_num_epoch_' + str(num_episodes) + '_targetUpdate_' + str(TARGET_UPDATE)

writer = ''
if writer_flag_on:
    writer = SummaryWriter(logdir='rl_data/' + model_name)

# ------------------------------------------------------------------------------------------------
#       Training LAM for whitebox + sample HDFS + DeepLog 
# ------------------------------------------------------------------------------------------------

# the anomaly dataset used to train the model
warmup_anomaly_buffers = ufp.read_from_file(warmup_data_name)

# the attack model
LAM = attack_model.LAM(n_input_features, n_hidden_size, n_layers, n_actions, window_size, state_size, model_name, EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, GAMMA)

random.seed(100)
sum_reward = 0
TP = 0

for i_episode in range(num_episodes):
    # randomly pick a sample from the anomaly
    rand_buffer_index = random.randint(0, len(warmup_anomaly_buffers) - 1)
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
    start_index = 0  # the position to start when creating the episode for greedy_policy implementation
    # --- debugging
    temp_batch_state = []
    temp_batch_changed_state = []
    temp_batch_next_state = []
    temp_batch_reward = []
    temp_batch_action = []
    temp_batch_greedy_Q_value = []
    temp_batch_modificaiton_type = []
    if debug_flag_on:
        print('episode: ', episode)
        print('modified buffer: ', modified_buffer)
        print('future logkeys: ', future_logkeys)
        print('---')
    # --------------
    for next_logkey in future_logkeys:
        # traverse through each new logkey and attack it
        if next_logkey == -1:
            # seperately handle termination state
            changed_state, next_state, action_taken, modification_flag, modified_buffer = LAM.attack(state, next_logkey, modified_buffer, False)
            next_state = None
            if changed_state[-1] == -1:
                changed_state.pop()
                modified_buffer.pop()
                changed_state = [modified_buffer[-state_size - 1]] + changed_state
            reward = LAM.reward_DeepLog(modification_flag, changed_state, DeepLog_model_params)
            greedy_policy_action_value = 0  # wont matter
            # ----- debugging
            if debug_flag_on:
                print(': ', next_logkey)
                print('state: ', state)
                print('next state: ', next_state)
                print('---')
                temp_batch_reward.append(reward)
                temp_batch_greedy_Q_value.append(greedy_policy_action_value)
                temp_batch_state.append(state)
                temp_batch_changed_state.append(changed_state)
                temp_batch_next_state.append(next_state)
                temp_batch_modificaiton_type.append(modification_flag)
                temp_batch_action.append(action_taken)  # cannot have a -1 indexed action
            # ---------------
            LAM.push_to_memory(state, action_taken, next_state, reward)
            sum_reward += reward
            break
        else:
            # obtain state transition and reward
            changed_state, next_state, action_taken, modification_flag, modified_buffer = LAM.attack(state, next_logkey, modified_buffer, False)
            rest_of_episode = future_logkeys[start_index + 1:len(future_logkeys)]  # the start_index logkey is already added into the modified buffer
            reward = LAM.reward_DeepLog(modification_flag, changed_state, DeepLog_model_params)
            # ---- debugging
            if debug_flag_on:
                print(': ', next_logkey)
                print('state: ', state)
                print('next_state: ', next_state)
                print('--')
                temp_batch_reward.append(reward)
                temp_batch_state.append(state)
                temp_batch_changed_state.append(changed_state)
                temp_batch_next_state.append(next_state)
                temp_batch_modificaiton_type.append(modification_flag)
                temp_batch_action.append(action_taken)
                temp_batch_greedy_Q_value.append(greedy_policy_action_value)
            # -------------
            LAM.push_to_memory(state, action_taken, next_state, reward)
            state = next_state.copy()
            sum_reward += reward
            start_index += 1
        loss = LAM.optimize_DQN()
        eps_threshold = LAM.get_eps_threshold()
        # add to tensorboardx
        if writer_flag_on:
            if loss is not None:
                writer.add_scalar('train_loss', loss.cpu(), i_episode + 1)
                writer.add_scalar('sum_reward', sum_reward, i_episode + 1)
    # update the target_net with policy_net in RL agent
    if i_episode % TARGET_UPDATE == 0:
        LAM.update_target_net()
    # save the policy network every 1000 epochs
    if i_episode % 1000 == 0:
        LAM.save_policy_net()
    # calculate the True Positive rate for warmup_data
    if i_episode % TP_TEST == 0:
        TP = 0
        advesarial_buffers, _ = LAM.scaled_attack(warmup_anomaly_buffers, DeepLog_model_params)
        for advesarial_buffer in advesarial_buffers:
            parsed_input = [int(x - 1) for x in advesarial_buffer]
            model_state = DeepLog_model_params[0]
            num_candidates = int(DeepLog_model_params[1])
            Anomaly, _ = DeepLog.flag_anomaly_in_buffer(model_state, parsed_input, window_size, n_input_features, num_candidates, 0)
            if Anomaly is True:
                TP += 1
        TP = TP / len(warmup_anomaly_buffers)
        if writer_flag_on:
            writer.add_scalar('TP_rate (warmup_data)', TP, i_episode + 1)
        else:
            print('i: ', i_episode, ' TP: ', TP)
    # ---- debugging
    if debug_flag_on:
        # debugging print statements
        print('sz state:', len(temp_batch_state))
        print('sz next_state:', len(temp_batch_next_state))
        print('sz changed_state: ', len(temp_batch_changed_state))
        # print('sz greedy_Q:', len(temp_batch_greedy_Q_value))
        print('sz actions: ', len(temp_batch_action))
        print('sz mod.made: ', len(temp_batch_modificaiton_type))
        print('sz rewards:', len(temp_batch_reward))
        # debugging print statements
        for i in range(len(temp_batch_reward)):
            s1 = temp_batch_state[i]
            s_ = temp_batch_changed_state[i]
            s2 = temp_batch_next_state[i]
            r = temp_batch_reward[i]
            a = temp_batch_action[i]
            print(i, ' ', s1, ' ', a, ' ', s_, ' -> ', s2, ' ', r)
        print('---------------------------------------------------------------------')

# save the model
LAM.save_policy_net()
if writer_flag_on:
    writer.close()

