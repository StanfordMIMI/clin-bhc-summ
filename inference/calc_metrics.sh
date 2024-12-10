#!/bin/bash
for i in 100
do 
    model=gpt4-chat
    case_id=$i
        
    python ../src/calc_metrics.py --model $model \
                            --case_id $case_id
done