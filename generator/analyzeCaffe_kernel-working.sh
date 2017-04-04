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
NVPROF_LOG_DIR='/home/sbchoi/git/gpu-cloud/generator/nvprof_log'
CAFFE_LOG='caffe_debug'

# making sure code runs on single machine, comment off next line it you want to use all gpus
export CUDA_VISIBLE_DEVICES='0'
mkdir -p $NVPROF_LOG_DIR
for entry in `ls $KERNEL_DIR`; do
  if [ "$entry" == $LAYER ] 
  then
    LIST=$LAYER
 fi
done 
kernels=`cat $KERNEL_DIR/$LIST`
for kernel in ${kernels}; do
  kernel_list=$kernel_list,$kernel
done
result_file=$LAYER-perf.txt
echo "analyzing ${kernel_list}, results will be stored in ${result_file}"

nvprof --csv --kernels ${kernel_list} --metrics ${PERF_ARGS} --log-file $NVPROF_LOG_DIR/${result_file} $CAFFE train --solver=$SOLVER  1>/dev/null 2>$CAFFE_LOG
