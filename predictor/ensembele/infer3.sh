#!/bin/bash

if [ -z "$1" ]
then 
echo "Please specify type of network you want to test"
echo "EX) InnerProduct, Convolution, Pooling"
exit
fi

if [ -z "$2" ]
then
echo "Please specify hidden number of neurons"
exit
fi 

PYTHON="/usr/bin/python" # dir to python
GPU_CLOUD_ROOT="/home/sbchoi/git/gpu-cloud"
DATA_DIR=$GPU_CLOUD_ROOT/predictor/data
#VAL_DIR=$DATA_DIR/"xor"
VAL_DIR=$DATA_DIR/"trn4_"$1
layer_type=$1
DATA_FILE="input_vec.txt"
LABEL_FILE="output_vec.txt"
TF="inferTFE3.py"

#FLAGS
LOG_FLAG='no'

#Logging
LOG_FILE=$layer_type"-log.txt"

#checkpointing
CHECK_DIR_ROOT="/home/sbchoi/git/gpu-cloud/predictor"

export CUDA_VISIBLE_DEVICES=`./free_gpu.sh`

ENSEMBLE_NUM=4 # number of ensembel models -1, so'4' means there are actually 5
ENSEMBLE_NUM_PARAM=$(($ENSEMBLE_NUM + 1))

if [ $LOG_FLAG == 'yes' ]
then
echo "Logging results to "$LOG_FILE
$PYTHON $TF $VAL_DIR/$DATA_FILE $VAL_DIR/$LABEL_FILE $VAL_DIR $ENSEMBLE_NUM_PARAM $layer_type $2 > $LOG_FILE

else
$PYTHON $TF $VAL_DIR/$DATA_FILE $VAL_DIR/$LABEL_FILE $VAL_DIR $ENSEMBLE_NUM_PARAM $layer_type $2 

fi