#!/bin/bash 

LAYERS=('Convolution' 'InnerProduct' 'Pooling' 'ReLU' 'SoftmaxWithLoss') 
LAYERS_ELEMENT=${#LAYERS[@]}

for (( h=0; h<${LAYERS_ELEMENT}; h++));
do
layer=${LAYERS[${h}]}

./train2.sh  $layer

done

