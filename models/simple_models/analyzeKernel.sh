#!/bin/bash
#Script for analyzing kernels

# code nees to be given as parameter
if [ -z "$1" ] 
then
 echo "Missing parameter Ex) ./analyzeCode example.py"
 exit 
fi

#some paramemeters need to be hardcoded 
CODE="$1"
PYTHON="/usr/bin/python"
LIST="kernel_list.txt" # list of the kernels to analyze
RESULT="kernel_perf_list.txt" # list of kernels and their performance

# serperate each metric by white space


#PERF_ARGS="sm_efficiency,ipc,dram_read_throughput,dram_write_throughput"
PERF_ARGS="all"

# nvprof --kernels SwapDimension --metrics sm_efficiency python train.py


# making sure code runs on single machine, comment off next line it you want to use all gpus
export CUDA_VISIBLE_DEVICES='0'
export TF_CPP_MIN_VLOG_LEVEL=0
#read kernels to analyze
kernels=`cat $LIST`

for kernel in ${kernels}; do
result_file="${kernel}_perf.txt"
echo "analyzing ${kernel}, results will be stored in ${result_file}"
nvprof -f --kernels ${kernel} --metrics ${PERF_ARGS} --log-file ${result_file} $PYTHON ${CODE} 1>/dev/null 2> /dev/null
done
