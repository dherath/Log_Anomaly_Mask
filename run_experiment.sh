#!/bin/bash

# note: the DeepLog models have already been pre-trained

python3 train_LAM.py 0.85 256 2000 4 128 10000 150 > results/trace_training.txt 2>&1 &
python3 test_LAM.py 0.85 256 2000 4 128 10000 150 > results/results_attack_effectiveness.txt 2>&1 &
