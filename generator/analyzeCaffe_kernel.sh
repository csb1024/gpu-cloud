#!/bin/bash

# script for tracing gpu kernels of caffe workloads
set -e 
if [ -z "$1" ] 
then
 echo "Missing solver prototxt parameter "
 exit
elif [ -z "$2" ]
then
 echo "Missing Layer paramemeter to profile, ** needs to match caffe layer catalogue's convention"
 exit 
fi

SOLVER="$1"
LAYER="$2"
CAFFE='/home/sbchoi/git/caffe/build/tools/caffe'
KERNEL_DIR='/home/sbchoi/git/gpu-cloud/generator/kernel_lists'
PERF_ARGS="dram_utilization,achieved_occupancy,sysmem_utilization" # string to be passed as argiment for --metric 

# making sure code runs on single machine, comment off next line it you want to use all gpus
export CUDA_VISIBLE_DEVICES='0'

for entry in `ls $KERNEL_DIR`; do
  if [ "$entry" == $LAYER ] 
  then
    LIST=$LAYER
 fi
done 
kernels=`cat $KERNEL_DIR/$LIST`
for kernel in ${kernels}; do
result_file=$kernel-perf.txt
echo "analyzing ${kernel}, results will be stored in ${result_file}"

nvprof --kernels ${kernel} --metrics ${PERF_ARGS} --log-file ${result_file} $CAFFE train --solver=$SOLVER  1>/dev/null 2> /dev/null
done
