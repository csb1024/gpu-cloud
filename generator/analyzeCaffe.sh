#!/bin/bash

# script for tracing gpu kernels of caffe workloads

if [ -z "$1" ] 
then
 echo "Missing parameter Ex) "
 exit 
fi
PYTHON="/usr/bin/python" # to ensure script executes vanilla python2.7 
CODE="$1"
NVPROF_LOG="out.txt"

LIST="kernel_list.txt"

# making sure code runs on single machine, comment off next line it you want to use all gpus
export CUDA_VISIBLE_DEVICES='0'

#making sure logging is disabled
export TF_CPP_MIN_VLOG_LEVEL=0

echo "Checking python $CODE"
$PYTHON ${CODE}

echo "Retrieving nvprof gpu_trace on ${NVPROF_LOG}"
nvprof --csv --log-file ${NVPROF_LOG} --print-gpu-trace  caffe 

# Extracts kernels names from NVPROF_LOG
awk -F "\"*,\"*" '{print $17}' ${NVPROF_LOG} > ${LIST}

#scp ${NVPROF_LOG} ${TF_VLOG_1} ${TF_VLOG_2} 143.248.139.71:~/
