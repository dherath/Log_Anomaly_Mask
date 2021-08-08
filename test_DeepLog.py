import torch
import torch.nn as nn
import time
import argparse

import os
import sys

#import util_bufferstream as bf
import util_fileprocessor as fp

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------
#              Hyperparameters
#------------------------------------------------------
# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 28
num_candidates = 9
num_epochs = 100
batch_size = 4096
name_flag = 'v1'

#------------------------------------------------------
#              Datset/Model-parameters
#-----------------------------------------------------

#- for the hdfs dataset, there are two filenames
filename_hdfs_abnormal = "data/hdfs_abnormal_test_complete_data"
filename_hdfs_normal = "data/hdfs_test_normal_reduced"
buffer_size = 15

buffer_stream_normal = []
buffer_blocknum_normal = []
buffer_stream_abnormal = []
buffer_blocknum_abnormal = []

# Note : for other datasets I think we will only need one buffer_stream

dataset_index = 1 # 1 - hdfs
model_path = str(os.getcwd())+'/model/'+'DeepLog_batch_size=' + str(batch_size) + '_epoch=' + str(num_epochs) +'_'+str(name_flag)+'.pt'

# generating the buffer stream dataset --------------------

if dataset_index == 1:
    #buffer_blocknum_normal, buffer_stream_normal = bf.bufferStreamHDFS(filename_hdfs_normal,buffer_size,window_size)
    #buffer_blocknum_abnormal, buffer_stream_abnormal = bf.bufferStreamHDFS(filename_hdfs_abnormal,buffer_size,window_size)
    stream_name = "HDFS"
elif dataset_index == 0:
    print("BUFFER-STREAM: incorrect data index")
    sys.exit(0)

#----------------------------------------------------------
#          Helper Functions
#----------------------------------------------------------

def generateHDFS(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            # hdfs.add(tuple(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

#def generateHDFS(buffer_stream_,stream_name):
#    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
#    #data = set()
#    data = []
#    for line in buffer_stream_:
#        line = list(map(lambda n: n , map(int, line)))
#        line = line + [-1] * (window_size + 1 - len(line)) #its window size + 1, the label is already there
#        #hdfs.add(tuple(line))
#        data.append(tuple(line))
#    print('+ Number of sessions({}): {}'.format(stream_name, len(data)))
#    return data
#
#---------------------------------------------------------
#       Model
#---------------------------------------------------------

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#--------------------------------------------------------
#        Model: Run
#--------------------------------------------------------

results_scores = ""
results_FN_seq = []
results_FP_seq = []
results_FN_label = []
results_FP_label = []

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    if dataset_index == 1:
        print('+ Predict : HDFS started')
        print('+ model_path: {}'.format(model_path))
        #buffer_stream_abnormal = bf.
        #test_normal_loader = generate(buffer_stream_normal,stream_name+'_normal' )
        #test_abnormal_loader = generate(buffer_stream_abnormal,stream_name+'_abnormal')
        test_normal_loader = generateHDFS(filename_hdfs_normal)
        test_abnormal_loader = generateHDFS(filename_hdfs_abnormal)
        #----------------------------------------------------------------------
        #        HDFS 
        #----------------------------------------------------------------------
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        # For normal HDFS -------------------
        #sz_normal_bufferstream = len(buffer_stream_normal)
        #completed_fraction = 0 #fraction of the buffers completed
        #index = 0 # for the block id
        #blocknum_FP = -1
        with torch.no_grad():
            print('+ HDFS: normal started')
            for line in test_normal_loader:
                #--------------------------------------
                # if current block is marked as anomaly (FP) then no need to check futher for same block
                #current_block_id = buffer_blocknum_normal[index]
                #index += 1
                #if blocknum_FP == current_block_id:
                #    continue
                #--------------------------------------
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    org_seq = list(map(lambda n: n+1, map(int,seq)))
                    org_label = label + 1
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        FP += 1
                        #blocknum_FP = current_block_id
                        results_FP_seq.append(org_seq)
                        results_FP_label.append(org_label)
                        #if completed_fraction >= 0.25:
                        #    print('+ FP saved')
                        #    fp.AppendListsOfListsToFile('0_results_FP_seq.txt',results_FP_seq)
                        #    fp.AppendListToFile('0_results_FP_label.txt',results_FP_label)
                        #    results_FP_seq = []
                        #    results_FP_label = []
                        #    completed_fraction = 0
                        break
            #completed_fraction += 1/sz_normal_bufferstream
        #fp.write_list_of_lists('0_results_FP_seq.txt',results_FP_seq)
        #fp.write_list('0_results_FP_label.txt',results_FP_label)
        print('+ HDFS: normal saved to file')
        results_FP_seq = []
        results_FP_label = []
        # For abnormal HDFS ------------------
        #sz_abnormal_bufferstream = len(buffer_stream_abnormal)
        #completed_fraction = 0  #fraction of abnormal_bufferstream completed
        #index = 0 #index for the block id
        #blocknum_TP = -1 # previous identified TP block
        with torch.no_grad():
            print('+ HDFS: abnormal started')
            for line in test_abnormal_loader:
                #---------------------------------------
                # if current block is marked as anomaly (TP) then dont check further
                #current_block_id - buffer_blocknum_abnormal[index]
                #index += 1
                #if blocknum_TP == current_block_id:
                #    continue
                #---------------------------------------
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    org_seq = list(map(lambda n: n+1, map(int,seq)))
                    org_label = label + 1
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        TP += 1
                        #blocknum_TP = current_block_id
                        break
                    else:
                        results_FN_seq.append(org_seq)
                        results_FN_label.append(org_label)
                        #if completed_fraction >= 0.25:
                        #    print('+ FN saved')
                        #    fp.AppendListsOfListsToFile('0_results_FN_seq.txt',results_FN_seq)
                        #    fp.AppendListToFile('0_results_FN_label.txt',results_FN_label)
                        #    results_FN_seq = []
                        #    results_FN_label = []
                        #    completed_fraction = 0
            #completed_fraction += 1/sz_abnormal_bufferstream
        #--------------------------------------
        # Compute precision, recall and F1-measure
        #print('+ prediction complete')
        FN = len(test_abnormal_loader) - TP
        TN = len(test_normal_loader) - FP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        #---------------------------------------
        TP_rate = TP / len(test_abnormal_loader)
        FN_rate = FN / len(test_abnormal_loader)
        FP_rate = FP / len(test_normal_loader)
        TN_rate = TN / len(test_normal_loader)
        results_scores =  ' TP: {:.5f}\n FN: {:.5f}\n FP: {:.5f}\n TN: {:.5f}\n'.format(TP_rate, FN_rate, FP_rate, TN_rate)
        results_scores += '-------------\n'
        results_scores += ' Precision: {:.5f}%\n Recall: {:.5f}%\n F1-measure: {:.5f}%'.format(P, R, F1)
        #print('NOTE : for calculation ')
        print('+ Predicted Results:')
        print(results_scores)
        #print('+ false positive (FP): {}, false negative (FN): {}, Precision: {:.5f}%, Recall: {:.5f}%, F1-measure: {:.5f}%'.format(FP, FN, P, R, F1))
        #print('+ Finished Predicting Precision/Recall use the actual TP/FN values')
        elapsed_time = time.time() - start_time
        print('+ elapsed_time: {}'.format(elapsed_time))
        #print('+ HDFS abnormal:  saving to file')
        #fp.WriteListsOfListsToFile('0_results_FP_seq.txt',results_FP_seq)
        #fp.WriteListsOfListsToFile('0_results_FP_label.txt',results_FP_label)
        #fp.write_list_of_lists('0_results_FN_seq.txt',results_FN_seq)
        #fp.write_list('0_results_FN_label.txt',results_FN_label)
        #f = open('0_results_scores.txt','w')
        #f.write(results_scores)
        #f.close()
        print('+ done')
    else:
        print('Invalid dataset_index')
        sys.exit(0)
