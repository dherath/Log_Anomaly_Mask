#!/bin/bash


# note: comment out models as needed
# DeepLog and LAM are already pre-trained

# 1. train DeepLog
# python3 train_DeepLog.py > results/trace_DeepLog_training.txt 2>&1

# 2. obtain True Positive rate before attack
# python3 test_DeepLog.py > results/results_before_attack.txt 2>&1

# 3. train LAM attack on sample anomaly buffers
# python3 train_LAM.py 0.85 256 2000 4 128 10000 150 > results/trace_training.txt 2>&1

# 4. perform attack and test effectivness
python3 test_LAM.py 0.85 256 2000 4 128 10000 150 > results/results_attack_effectiveness.txt 2>&1 &
