#!/bin/bash

# script for tracing gpu kernels of caffe workloads

if [ -z "$1" ] 
then
 echo "Missing parameter Ex) ./analyzeCaffe /dir/to/your/solver.prototxt"
 exit 
fi

SOLVER="$1"
NVPROF_LOG="out.txt"
LIST="kernel_list.txt"
LIST2="kernel_list_dedup.txt"
CAFFE_ROOT='/home/sbchoi/git/caffe'

#Automatically dedup kernel names? change to yes if you want to dedup
DEDUP_KERNEL="yes"

# making sure code runs on single machine, comment off next line it you want to use all gpus
export CUDA_VISIBLE_DEVICES='0'

echo "Retrieving nvprof gpu_trace on ${NVPROF_LOG} for training"
nvprof --csv --log-file ${NVPROF_LOG} --print-gpu-trace  $CAFFE_ROOT/build/tools/caffe train --solver=$SOLVER

# Extracts kernels names from NVPROF_LOG
awk -F "\"*,\"*" '{print $17}' ${NVPROF_LOG} > ${LIST}

#scp ${NVPROF_LOG} ${TF_VLOG_1} ${TF_VLOG_2} 143.248.139.71:~/

if [ $DEDUP_KERNEL == "yes" ]
then
  sort $LIST | uniq > $LIST2
fi

