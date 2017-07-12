#!/bin/bash

if [ -z "$1" ]
then 
echo "Please specify type of network you want to train"
echo "EX) InnerProduct, Convolution, Pooling"
exit
fi

PYTHON="/usr/bin/python" # dir to python
TRN_DIR="trn3_"$1
VAL_DIR="trn3_"$1
DATA_FILE="input_vec.txt"
LABEL_FILE="output_vec.txt"
TF="execTFMLP2.py"

#FLAGS
LOG_FLAG='yes'
RESTORE_FLAG='no'

#Logging
LOG_FILE="log.txt"

#checkpointing
CHECK_DIR_ROOT="/home/sbchoi/git/gpu-cloud/predictor"
mkdir -p $CHECK_DIR_ROOT/$1-checkpoint
CHECK_DIR=$CHECK_DIR_ROOT/$1-checkpoint 

if [ $RESTORE_FLAG  == "yes" ] 
then
CHECK_DIR_PARAM=$CHECK_DIR
else
CHECK_DIR_PARAM="new"
fi

if [ $LOG_FLAG == "no" ]
then
echo "Logging results to "$LOG_FILE
$PYTHON $TF $TRN_DIR/$DATA_FILE $TRN_DIR/$LABEL_FILE $VAL_DIR/$DATA_FILE $VAL_DIR/$LABEL_FILE $CHECK_DIR_PARAM > $LOG_FILE

else
$PYTHON $TF $TRN_DIR/$DATA_FILE $TRN_DIR/$LABEL_FILE $VAL_DIR/$DATA_FILE $VAL_DIR/$LABEL_FILE $CHECK_DIR_PARAM
fi

mv $CHECK_DIR_ROOT/model.ckpt* $CHECK_DIR
mv $CHECK_DIR_ROOT/checkpoint $CHECK_DIR
