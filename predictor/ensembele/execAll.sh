#!/bin/bash

TRAIN_FLAG='yes' # do training ?
TEST_FLAG='yes' # proceed with testing?

PYTHON="/usr/bin/python"
ANALYZER="analyzeResults.py"
#LAYERS=('Convolution' 'InnerProduct' 'Pooling' 'ReLU' 'SoftmaxWithLoss') 
#LAYERS=('SoftmaxWithLoss')
#LAYERS=('ReLU')
LAYERS=('Pooling')
LAYERS_ELEMENT=${#LAYERS[@]}

for (( h=0; h<${LAYERS_ELEMENT}; h++));
do
layer=${LAYERS[${h}]}

if [ $TRAIN_FLAG == 'yes' ]
then
./train.sh $layer
fi

if [ $TEST_FLAG == 'yes' ]
then
# perform inference
./infer.sh $layer
mkdir -p $layer
mv new_output_vec.txt $layer/
mv output_vec*.txt $layer/
mv input_vec.txt $layer/
mv label_vec.txt $layer/

$PYTHON $ANALYZER $layer/new_output_vec.txt $layer/label_vec.txt $layer/analyzed_results.txt

fi



done
