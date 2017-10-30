#!/bin/bash

if [ -z "$1" ]
then 
echo "Please specify type of network you want to train"
echo "EX) InnerProduct, Convolution, Pooling"
exit
fi

PYTHON="/usr/bin/python" # dir to python
DATA_DIR="/home/sbchoi/git/gpu-cloud/predictor/data"
TRN_DIR=$DATA_DIR/time/"trnF_"$1
VAL_DIR=$DATA_DIR/time/"trnF_"$1

#TRN_DIR=$DATA_DIR/"xor"
#VAL_DIR=$DATA_DIR/"xor"
DATA_FILE="input_vec.txt"
LABEL_FILE="output_vec.txt"
TF="train_v5.py"
ANALYZER="analyzeResults2.py"

#FLAGS
LOG_FLAG='yes'

#Logging

#checkpointing
CHECK_DIR_ROOT="/home/sbchoi/git/gpu-cloud/predictor/checkpoints"

ENSEMBLE_NUM=0 # number of ensembel models -1 => if 4, there are actually 5 models


#hyper parameters we are going to try out
BATCHS=( '256')
#BATCHS=('4')
INIT_LRS=( '1' '2')
#INIT_LRS=('0.01')
H_NUMS=( '40' '60' '80' '100')
#H_NUMS=('5')
MAX_EPOCHS=('500000')
#MAX_EPOCHS=('5000')
BATCHS_ELEMENT=${#BATCHS[@]} 
INIT_LRS_ELEMENT=${#INIT_LRS[@]} 
H_NUMS_ELEMENT=${#H_NUMS[@]} 
MAX_EPOCHS_ELEMENT=${#MAX_EPOCHS[@]} 
##
#TRIAL_NUM, in order to keep track on which training phase we are on
TRIAL_NUM=1
export CUDA_VISIBLE_DEVICES=`./free_gpu.sh`
#GPU_ID=$CUDA_VISIBLE_DEVICES
GPU_ID=$CUDA_VISIBLE_DEVICES
LOG_DIR=$PWD/gpu$GPU_ID
GPU_DIR=$LOG_DIR

echo "Batch,initial learning rate,num of nidden neurons, maximum epochs " > $LOG_DIR/config_list.txt
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
CHECK_DIR=$CHECK_DIR_ROOT/$1-checkpoint-forward-$i
mkdir -p $CHECK_DIR
CHECK_DIR_PARAM=$CHECK_DIR_ROOT
if [ $LOG_FLAG == "yes" ]
then
LOG_FILE=$i-log.txt
echo "Logging results to "$LOG_DIR"/"$LOG_FILE
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM $init_lr $max_epoch $batch $h_num $GPU_DIR > $LOG_DIR/$LOG_FILE 

else
echo "Start training"
$PYTHON $TF $TRN_DIR $VAL_DIR $CHECK_DIR_PARAM $init_lr $max_epoch $batch $h_num $GPU_DIR
fi

mv $CHECK_DIR_ROOT/model.ckpt* $CHECK_DIR
mv $CHECK_DIR_ROOT/checkpoint $CHECK_DIR
cp $LOG_DIR/input_vec.txt $CHECK_DIR
cp $LOG_DIR/output_vec.txt $CHECK_DIR
cp $LOG_DIR/label_vec.txt $CHECK_DIR
done # ENSEMBLE_NUM

./infer_v5.sh $1 $h_num $GPU_ID

#mkdir -p $1
#cp $LOG_DIR/new_output_vec.txt $1/
#cp $LOG_DIR/output_vec*.txt $1/
#cp $LOG_DIR/input_vec.txt $1/
#cp $LOG_DIR/label_vec.txt $1/

$PYTHON $ANALYZER $LOG_DIR/new_output_vec.txt $LOG_DIR/label_vec.txt $LOG_DIR/analyzed_results.txt 

mv $LOG_DIR/analyzed_results.txt $LOG_DIR/$TRIAL_NUM-results.txt
echo $batch","$init_lr","$h_num","$max_epoch >> $LOG_DIR/config_list.txt
TRIAL_NUM=$((TRIAL_NUM + 1))
done #MAX_EPOCHS
done #N_NUMS
done #INIT_LRS
done #BATCHS
