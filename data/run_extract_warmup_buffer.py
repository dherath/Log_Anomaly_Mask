import random
import util_fileprocessor as fp

def extract_abnormal_data(filename):
    abnormal_small = []
    abnormal_complete = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            if line != []:
              abnormal_complete.append(line)
    return abnormal_complete

data = extract_abnormal_data('hdfs_abnormal_test_complete_data')
tot_samples = len(data)
warmup_size = 200
index = random.sample(range(tot_samples),warmup_size)

warmup_data = []
for i in range(warmup_size):
    warmup_data.append(data[index[i]])

fp.write_list_of_lists('warmup_data',warmup_data)
