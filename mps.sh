#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 # select GPU

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# next is unnecessary for single user system
#nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set GPU 0 to exclusive mode
nvidia-cuda-mps-control -d 

