#!/bin/bash 

LAYERS=('Convolution' 'InnerProduct' 'Pooling' 'ReLU' 'LSTM') 
LAYERS_ELEMENT=${#LAYERS[@]}

for (( h=0; h<${LAYERS_ELEMENT}; h++));
do
layer=${LAYERS[${h}]}

./train_v5.sh  $layer

done

