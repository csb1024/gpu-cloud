#!/bin/bash

MON='nvml_mon'
PYTHON='/usr/bin/python'
export CUDA_VISIBLE_DEVICES='0'

for i in $(seq 1 30) 
do
	./${MON} ${PWD} &
	hidden_num=$(( i*10))
	$PYTHON multilayer_perceptron.py $hidden_num $hidden_num >> duration.txt 
	mv monitor_log.txt log_${hidden_num}.txt
done
