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

#TRN_DIR=$DATA_DIR/"xor"
#VAL_DIR=$DATA_DIR/"xor"
DATA_FILE="input_vec.txt"
LABEL_FILE="output_vec.txt"
TF="trainTFE2.py"
ANALYZER="analyzeResults2.py"

#FLAGS
LOG_FLAG='yes'
RESTORE_FLAG='no'

#Logging

#checkpointing
CHECK_DIR_ROOT="/home/sbchoi/git/gpu-cloud/predictor"

ENSEMBLE_NUM=4 # number of ensembel models -1 => if 4, there are actually 5 models


#hyper parameters we are going to try out
BATCHS=('64')
#BATCHS=('4')
INIT_LRS=( '0.5')
#INIT_LRS=('0.01')
H_NUMS=( '25')
#H_NUMS=('5')
MAX_EPOCHS=('200000')
#MAX_EPOCHS=('5000')
BATCHS_ELEMENT=${#BATCHS[@]} 
INIT_LRS_ELEMENT=${#INIT_LRS[@]} 
H_NUMS_ELEMENT=${#H_NUMS[@]} 
MAX_EPOCHS_ELEMENT=${#MAX_EPOCHS[@]} 
##
#TRIAL_NUM, in order to keep track on which training phase we are on
TRIAL_NUM=1



echo "Batch,initial learning rate,num of nidden neurons, maximum epochs " > config_list.txt
for (( a=0; a<${BATCHS_ELEMENT}; a++));do

batch=${BATCHS[${a}]}
for (( b=0; b<${INIT_LRS_ELEMENT}; b++));do
init_lr=${INIT_LRS[${b}]}

for (( c=0; c<${H_NUMS_ELEMENT}; c++));do
h_num=${H_NUMS[${c}]}

for (( d=0; d<${MAX_EPOCHS_ELEMENT}; d++));do
max_epoch=${MAX_EPOCHS[${d}]}

echo "Testing Trial : "$TRIAL_NUM
for i in `seq 0 $ENSEMBLE_NUM` 
do
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
LOG_FILE=$i-log.txt
echo "Logging results to "$LOG_FILE
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM $init_lr $max_epoch $batch $h_num > $LOG_FILE 

else
echo "Start training"
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM $init_lr $max_epoch $batch $h_num 
fi

mv $CHECK_DIR_ROOT/model.ckpt* $CHECK_DIR
mv $CHECK_DIR_ROOT/checkpoint $CHECK_DIR
mv $PWD/input_vec.txt $CHECK_DIR
mv $PWD/output_vec.txt $CHECK_DIR
mv $PWD/label_vec.txt $CHECK_DIR
done # ENSEMBLE_NUM

./infer2.sh $1 $h_num

mkdir -p $1
mv new_output_vec.txt $1/
mv output_vec*.txt $1/
mv input_vec.txt $1/
mv label_vec.txt $1/

$PYTHON $ANALYZER $1/new_output_vec.txt $1/label_vec.txt $1/analyzed_results.txt 

cp $1/analyzed_results.txt temp.txt
mv temp.txt $TRIAL_NUM-results.txt 
echo $batch","$init_lr","$h_num","$max_epoch >> config_list.txt
TRIAL_NUM=$((TRIAL_NUM + 1))
done #MAX_EPOCHS
done #N_NUMS
done #INIT_LRS
done #BATCHS
