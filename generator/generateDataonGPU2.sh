#!/bin/bash 

# The ultimate script... data generator

if [ -z "$1" ]
then
echo "PLease specicfy the number of random training data you want to make"
exit
fi

if [ -z "$2" ]
then
echo "Please specify the gpu number you want to train on * Choose between 0~3"
exit
fi

NUM_DATA=$1
GPU_ID=$2
#LAYERS=('Convolution' 'InnerProduct' 'Pooling' 'ReLU' 'SoftmaxWithLoss' 'LRN' 'Dropout' 'Data') #planning to expand
#LAYER_NAME=('CONV' 'IP' 'POOLING' 'RELU' 'SOFTMAX' 'LRN' 'DROP' 'IMAGEDATA') # MUST maintain the same order to the corressponding layer on top list
#LAYERS=('LRN' 'Dropout' 'Data')
LAYERS=('LSTM')
#LAYER_NAME=('LRN' 'DROP' 'IMAGEDATA')
LAYER_NAME=('LSTM')
LAYERS_ELEMENT=${#LAYERS[@]}

#Here are important DIRs
PYTHON='/usr/bin/python'
GPU_CLOUD_DIR='/home/sbchoi/git/gpu-cloud'
GENERATOR_ROOT_DIR=$GPU_CLOUD_DIR/generator
GENERATOR_DIR=$GENERATOR_ROOT_DIR/gpu"$GPU_ID"
STORE_DIR=$GPU_CLOUD_DIR/backup/gpu"$GPU_ID"
#DIR related to input
GENERATOR='genPrototxt.py'
BASE_PROTOTXT='dummy.prototxt'
RAND_PROTOTXT='out.prototxt'
INPUT_VEC="input_vec.txt"

# DIR related to output 
CAFFE_LOG_DIR=$GENERATOR_DIR
NVPROF_LOG_DIR=$GENERATOR_DIR/nvprof_log
CAFFE_LOG='caffe_debug'
OUTPUT_VEC="output_vec.txt"
LOG_PARSER='parseOutput.py'
#solver prototxt 

# for maintainance

for (( h=0; h<${LAYERS_ELEMENT}; h++));
do
layer=${LAYERS[${h}]}
layer_name=${LAYER_NAME[${h}]}
BASE_DIR=$PWD/layer/$layer
if [ "$layer" == "LSTM" ]
then
GENERATOR='genPrototxt_lstm.py'
fi 

i=1
while [ $i -le ${NUM_DATA} ] ; do

$PYTHON $GENERATOR_ROOT_DIR/$GENERATOR  $BASE_DIR/$BASE_PROTOTXT $GENERATOR_DIR/$RAND_PROTOTXT $layer $GENERATOR_DIR/$INPUT_VEC 


./analyzeCaffe_kernel.sh $GENERATOR_DIR/data_generator.prototxt $layer $NVPROF_LOG_DIR $CAFFE_LOG_DIR $GPU_ID


$PYTHON $GENERATOR_ROOT_DIR/$LOG_PARSER $CAFFE_LOG_DIR/$CAFFE_LOG $layer_name $layer $GENERATOR_DIR/$OUTPUT_VEC $NVPROF_LOG_DIR

echo "data number "$i" has been generated on GPU"$GPU_ID" congrats!"
((i++))

done
mv $GENERATOR_DIR/$INPUT_VEC $STORE_DIR/$layer_name-input_vec.txt
mv $GENERATOR_DIR/$OUTPUT_VEC $STORE_DIR/$layer_name-output_vec.txt
done
