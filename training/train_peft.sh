#!/bin/bash

model=flan-ul2
case_id=200

python ../src/train_peft.py --model $model \
                         --case_id $case_id