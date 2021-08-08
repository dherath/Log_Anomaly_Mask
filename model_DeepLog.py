import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------
#                       MODEL : DEEPLOG
# ------------------------------------------------------------------------


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

# ------------------------------------------------------------------------
#                    DEEPLOG FUNCTIONS
# ------------------------------------------------------------------------


def generate(name, window_size):
    """
    function to load the dataset of buffers
    """
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    # hdfs = set()
    hdfs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            # hdfs.add(tuple(line))
            hdfs.append(list(line))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


def load_ensemble(path, versions, specification):
    """
    loads an ensemble of deeplog models and returns them
    ----
    param path : str(os.getcwd())+'/model/DeepLog_batch_size=4096_epoch=100_' or the path to the models
    param versions : list ['v1','v2','v3',...] trained models
    param specification : list [input_size=1,hidden_size,num_layers,num_classes,num_candidates]
    ----
    return models : list [model_v1,model_v2,model_v3,...] loaded DeepLog models
    """
    models = []
    input_size = specification[0]
    hidden_size = specification[1]
    num_layers = specification[2]
    num_classes = specification[3]
    for version in versions:
        model_path = str(path) + str(version) + '.pt'
        temp_model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
        temp_model.load_state_dict(torch.load(model_path, map_location=device))
        temp_model.eval()
        models.append(temp_model)
    return models


def flag_anomaly_in_buffer(model, test_buffer, window_size, input_size, num_candidates, start_index):
    """
    returns the index the moment an anomaly is flagged in buffer
    this is done to reduce overlap of multiple anomaly flags
    ----
    param model : the DeepLog model
    param test_buffer : the test sample
    param window_size : m of the model
    param input_size : 1
    param num_candidates : the top g candidates to choose from
    param start_index : the index to start creating sequences
    ----
    return anomaly_flag : True if Anomaly was caught, False otherwise
    return seq_index : the sequence index where the anomaly was caught (always >= start_index)
    """
    anomaly_flag = False
    seq_index = -1
    for i in range(start_index, len(test_buffer) - window_size):
        seq = test_buffer[i:i + window_size]  # seperate the original seq,label pair
        label = test_buffer[i + window_size]
        torch_seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        torch_label = torch.tensor(label).view(-1).to(device)
        output = model(torch_seq)  # calling the model and getting predictions
        predicted = torch.argsort(output, 1)[0][-num_candidates:]
        # if the prediction is not within top g=num_candidates, then flag anomaly 
        if torch_label not in predicted:
            anomaly_flag = True
            seq_index = i
            break
    return anomaly_flag, seq_index


def flag_anomaly_in_seq(model, test_seq, test_label, window_size, input_size, num_candidates):
    """
    flags if asequence is an anomaly or not
    ----
    param model:
    param test_seq:
    param test_label:
    param window_size:
    param input_size:
    param num_candidates:
    ---
    return True if Anomaly else False
    """
    torch_seq = torch.tensor(test_seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
    torch_label = torch.tensor(test_label).view(-1).to(device)
    # call the model and get predictions
    output = model(torch_seq)  # calling the model and getting predictions
    predicted = torch.argsort(output, 1)[0][-num_candidates:]
    if torch_label not in predicted:
        return True
    return False
