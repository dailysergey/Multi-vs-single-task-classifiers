#!/bin/bash

[ -e results.csv ] && rm results.csv
for task in "mrpc" "cola" "sst2" 
do
    for seed in 1 2 3
    do
        for model in "roberta-base"
        do
            python3 transformer_glue.py --task $task --model $model --seed $seed --epochs 2 \
                                        --log_file results.csv "$@"
        done
    done
done