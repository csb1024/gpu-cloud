#!/bin/bash

if [ -z "$1" ]
then
echo "Please specify GPU number" 
exit
fi  
ROOT_DIR=/home/sbchoi/git/gpu-cloud

MONITOR_DIR=$ROOT_DIR/monitor/cpp3

NVML_MON=$MONITOR_DIR/nvml_mon

GEN_DIR=$ROOT_DIR/generator

GPU_DIR=$GEN_DIR/gpu$1

#run nvml_mon in background

$NVML_MON $GEN_DIR  $1 &
#execute Caffe
CAFFE='/home/sbchoi/git/caffe/build/tools/caffe'

sleep 1

$CAFFE time -model $GPU_DIR/out.prototxt -iterations 10 2>$GPU_DIR/caffe_debug -gpu $1 


echo "Results will be store in $GEN_DIR " 
