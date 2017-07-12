#!/bin/bash


# According to mps docs, GPU uuid should be used
#export CUDA_VISIBLE_DEVICES="GPU-7fa81e95,GPU-5fe81606,GPU-e0ea6c7b,GPU-c9f9421d"
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# Number of gpus with compute_capability 3.5  per server
NGPUS=4

# Start the MPS server for each GPU
for ((i=0; i< $NGPUS; i++))

do
 mkdir /tmp/mps_$i
 mkdir /tmp/mps_log_$i
 export CUDA_VISIBLE_DEVICES=$i
 export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
 export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i
 nvidia-cuda-mps-control -d
done

#export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

#export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# next is unnecessary for single user system
#nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set GPU 0 to exclusive mode
#nvidia-cuda-mps-control -d 

