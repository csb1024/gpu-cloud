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

if [ -z "$6" ]
then 
echo "Missing profiling parameter "
exit
fi

SOLVER="$1"
LAYER="$2"
CAFFE='/home/sbchoi/git/caffe/build/tools/caffe'
ROOT_DIR='/home/sbchoi/git/gpu-cloud'
MODEL_dIR=$ROOT_DIR/generator/layer/$LAYER
KERNEL_DIR='/home/sbchoi/git/gpu-cloud/generator/kernel_lists'
#PERF_ARGS="dram_utilization,achieved_occupancy,sysmem_utilization" # string to be passed as argiment for --metric 
#PERF_ARGS="sm_activity,ipc,dram_utilization,achieved_occupancy,gld_throughput,gst_throughput"
PERF_ARGS="sm_activity"

NVPROF_LOG_DIR=$3
CAFFE_LOG_DIR=$4
GPU_ID=$5
CAFFE_LOG='caffe_debug'
profile=$6
# making sure code runs on single machine, comment off next line it you want to use all gpus
#export CUDA_VISIBLE_DEVICES='0'
mkdir -p $NVPROF_LOG_DIR
#rm $NVPROF_LOG_DIR/*
result_file="nvprof_log.csv"

#erase all files before profile
#rm $NVPROF_LOG_DIR/*

#echo "run training code"
#nvprof --profile-from-start off --metrics $PERF_ARGS --csv --log-file $NVPROF_LOG_DIR/$result_file $CAFFE train --solver $SOLVER  -gpu $GPU_ID -profile $profile 1>/dev/null 2>$CAFFE_LOG_DIR/$CAFFE_LOG
#echo "run testing code"
#nvprof --profile-from-start-off --metrics ${PERF_ARGS} --csv --log-file $NVPROF_LOG_DIR/${result_file} $CAFFE test -model $CAFFE_LGG_DIR/out.prototxt -iterations 1 -gpu $GPU_ID -profile $profile 1>/dev/null 2>$CAFFE_LOG_DIR/$CAFFE_LOG







#nvprof profile run
echo "run training code"
nvprof --profile-from-start off --metrics $PERF_ARGS --csv --log-file $NVPROF_LOG_DIR/$result_file $CAFFE train --solver $SOLVER  -gpu $GPU_ID -profile $profile 1>/dev/null 2>$CAFFE_LOG_DIR/$CAFFE_LOG

#nvprof --profile-from-start off --metrics $PERF_ARGS --csv --log-file $NVPROF_LOG_DIR/$result_file $CAFFE train --solver $SOLVER  -gpu $GPU_ID -profile $profile


#echo "runinng training code again"
# for accurate time measurement, we execute caffe one more time
