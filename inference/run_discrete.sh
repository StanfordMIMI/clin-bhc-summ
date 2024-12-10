#!/bin/bash

model=llama2-13b-chat
case_id=0

python ../src/run_discrete.py --model $model \
                           --case_id $case_id