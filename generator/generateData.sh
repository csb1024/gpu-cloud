#!/bin/bash 

# The ultimate script... data generator

if [ -z "$1" ]
then
echo "PLease specicfy the number of random training data you want to make"
exit
fi

#LAYERS=('Convolution' 'InnerProduct' 'Pooling') #planning to expand
#LAYER_NAME=('CONV' 'IP' 'POOLING') # MUST maintain the same order to the corressponding layer on top list
LAYERS=('Convolution' 'InnerProduct')
LAYER_NAME=( 'CONV' 'IP')
LAYERS_ELEMENT=${#LAYERS[@]}

#Here are important DIRs
PYTHON='/usr/bin/python'
GPU_CLOUD_DIR='/home/sbchoi/git/gpu-cloud'
GENERATOR_DIR=$GPU_CLOUD_DIR/generator
STORE_DIR=$GPU_CLOUD_DIR/backup
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



NUM_DATA=$1

# for maintainance
mv $GENERATOR_DIR/*_vec.txt $STORE_DIR
cp $GENERATOR_DIR/$GENERATOR $GENERATOR_DIR/$GENERATOR-working

for (( h=0; h<${LAYERS_ELEMENT}; h++));
do
layer=${LAYERS[${h}]}
layer_name=${LAYER_NAME[${h}]}
i=1
while [ $i -le ${NUM_DATA} ] ; do

$PYTHON $GENERATOR_DIR/$GENERATOR $GENERATOR_DIR/$GENERATOR_$BASE_PROTOTXT $GENERATOR_DIR/$RAND_PROTOTXT $layer $GENERATOR_DIR/$INPUT_VEC 
./analyzeCaffe_kernel.sh $GENERATOR_DIR/data_generator.prototxt $layer
$PYTHON $GENERATOR_DIR/$LOG_PARSER $CAFFE_LOG_DIR/$CAFFE_LOG $layer_name $layer $OUTPUT_VEC
((i++))
done
mv $INPUT_VEC $STORE_DIR/$layer_name-input_vec.txt
mv $OUTPUT_VEC $STORE_DIR/$layer_name-output_vec.txt
done
