#!/bin/bash

for ((i=0;i<3;i+=1))
do
	python main.py \ 
        --env_id="HalfCheetah-v5" \
        --seed=$i

    python main.py \ 
        --env_id="Ant-v5" \
        --seed=$i

    python main.py \ 
        --env_id="Walker2d-v5" \
        --seed=$i

    python main.py \ 
        --env_id="Hopper-v5" \
        --seed=$i
done