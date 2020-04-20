### Dataset Information

#### Data

Only use `Test/Train/Attack` datasets

1. hdfs_abnormal_test_complete_data : `Test` dataset for abnormal/anomaly behaviour
2. hdfs_abnormal_test_small_data : `X` not used, small abnormal sequences
3. hdfs_test_abnormal : **original** abnormal data
4. hdfs_test_normal : **original** normal data
5. hdfs_test_normal_reduced : `Test` dataset for normal behaviour
6. hdfs_train : **original** train data for DeepLog (only normal behaviour)
7. hdfs_train_increased : `Train` data for DeepLog (normal data)
8. warmup_data : `Attack` data (abnormal behaviour) for **RL attack** algorithms

#### Code

+ run_extract_data.py : extract abnormal complete/small from __hdfs_test_abnormal__
+ run_extract_warmup_buffer.py : extract the warmup buffers (removes it from test dataset)
+ util_fileprocessor.py : util__ code