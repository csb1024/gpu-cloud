#!/bin/bash

if [ -z "$1" ]
then 
echo "Please specify type of network you want to train"
echo "EX) InnerProduct, Convolution, Pooling"
exit
fi

PYTHON="/usr/bin/python" # dir to python
DATA_DIR="/home/sbchoi/git/gpu-cloud/predictor/data"
TRN_DIR=$DATA_DIR/"trn4_"$1
VAL_DIR=$DATA_DIR/"trn4_"$1
DATA_FILE="input_vec.txt"
LABEL_FILE="output_vec.txt"
TF="trainUnnorm.py"
ANALYZER="analyzeResults.py"

#FLAGS
LOG_FLAG='no'
RESTORE_FLAG='no'

#Logging
LOG_FILE="log.txt"

#checkpointing
CHECK_DIR_ROOT="/home/sbchoi/git/gpu-cloud/predictor"

ENSEMBLE_NUM=4 # number of ensembel models -1 => if 4, there are actually 5 models

i="0"
CHECK_DIR=$CHECK_DIR_ROOT/$1-checkpoint-$i
mkdir -p $CHECK_DIR

if [ $RESTORE_FLAG  == "yes" ] 
then
CHECK_DIR_PARAM=$CHECK_DIR
else
CHECK_DIR_PARAM="new"
fi



if [ $LOG_FLAG == "yes" ]
then
echo "Logging results to "$LOG_FILE
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM > $LOG_FILE

else
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM
fi

mv $CHECK_DIR_ROOT/model.ckpt* $CHECK_DIR
mv $CHECK_DIR_ROOT/checkpoint $CHECK_DIR

$PYTHON $ANALYZER $PWD/output_vec.txt $PWD/label_vec.txt $PWD/analyzed_results.txt 

