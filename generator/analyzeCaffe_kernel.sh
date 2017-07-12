#!/bin/bash

# script for tracing gpu kernels of caffe workloads
set -e 
if [ -z "$1" ] 
then
 echo "Missing solver prototxt parameter "
 exit
fi
if [ -z "$2" ]
then
 echo "Missing Layer paramemeter to profile, ** needs to match caffe layer catalogue's convention"
 exit 
fi

if [ -z "$3" ]
then
echo "Missing parameter which states the directory for storing nvprof logs "
exit
fi

if [ -z "$4" ]
then
echo "Missing parameter which states the directory for storing caffe debug log"
exit
fi

if [ -z "$5" ]
then
echo "Missing parameter for GPU ID"
exit
fi


SOLVER="$1"
LAYER="$2"
CAFFE='/home/sbchoi/git/caffe/build/tools/caffe'
KERNEL_DIR='/home/sbchoi/git/gpu-cloud/generator/kernel_lists'
PERF_ARGS="dram_utilization,achieved_occupancy,sysmem_utilization,executed_ipc" # string to be passed as argiment for --metric 
NVPROF_LOG_DIR=$3
CAFFE_LOG_DIR=$4
GPU_ID=$5
CAFFE_LOG='caffe_debug'
SKIP_KERNELS="yes"

# making sure code runs on single machine, comment off next line it you want to use all gpus
#export CUDA_VISIBLE_DEVICES='0'
mkdir -p $NVPROF_LOG_DIR

if [ $SKIP_KERNELS == "yes" ];then
$CAFFE time -model $CAFFE_LOG_DIR/out.prototxt -iterations 1 2>$CAFFE_LOG_DIR/$CAFFE_LOG -gpu $GPU_ID
else

for entry in `ls $KERNEL_DIR`; do
  if [ "$entry" == $LAYER ] ; then
    LIST=$LAYER
 fi
done 
kernels=`cat $KERNEL_DIR/$LIST`
for kernel in ${kernels}; do
result_file=$kernel-perf.txt
echo "analyzing ${kernel}, results will be stored in ${result_file}"

nvprof --csv --kernels ${kernel} --metrics ${PERF_ARGS} --log-file $NVPROF_LOG_DIR/${result_file} $CAFFE train --solver=$SOLVER  1>/dev/null 2>$CAFFE_LOG_DIR/$CAFFE_LOG -gpu $GPU_ID
done

fi
