#!/bin/bash

PYTHON='/usr/bin/python'
export CUDA_VISIBLE_DEVICES='0'

for i in $(seq 1 10) 
do
	hidden_num=$(( i*10))
	$PYTHON multilayer_perceptron.py $hidden_num $hidden_num >> duration.txt 
done
