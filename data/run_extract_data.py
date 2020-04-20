import util_fileprocessor as fp

def extract_abnormal_data(filename,window_size):
    abnormal_small = []
    abnormal_complete = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            #num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            if len(line) <= window_size:
                abnormal_small.append(line)
            else:
                abnormal_complete.append(line)
    return abnormal_small,abnormal_complete


#------------------------------------------------------------------------
# TEST PHASE
#------------------------------------------------------------------------

print("+started")
filename = '../data/hdfs_test_abnormal'
window_size = 10
abnormal_small,abnormal_complete = extract_abnormal_data(filename,window_size)
fp.writeListsOfListsToFile('hdfs_abnormal_test_small_data',abnormal_small)
fp.writeListsOfListsToFile('hdfs_abnormal_test_complete_data',abnormal_complete)
print("+complete")
