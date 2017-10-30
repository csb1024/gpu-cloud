#!/bin/bash 


if [ -z "$1" ]
then
echo "please specify prefix Ex) CONV, POOLING, IP"
exit
fi

if [ -z "$2" ]
then
echo "Please specifiy the root dircory of data"
exit
fi 

if [ -z "$3" ]
then
echo "Please specify 'forward' or 'backward'"
exit
fi

if [ -z "$4" ]
then
echo "Specify gpuid"
exit
fi

if [ "$1" == "CONV" ]
then
layerType="Convolution" 
elif [ "$1" == "IP" ]
then
layerType="InnerProduct" 
elif [ "$1" == "RELU" ]
then
layerType="ReLU" 
elif [ "$1" == "POOLING" ]
then
layerType="Pooling" 
elif [ "$1" == "LSTM" ]
then
layerType="LSTM" 
else
echo "Please enter valid prefix"
exit 
fi


if [ "$3" == "forward" ]
then
phase="F"
elif [ "$3" == "backward" ]
then
phase="B"
else
echo "Please specifiy forward or backward"
exit
fi

gpuid=$4
./filterAllData.sh $1
DATA_ROOT_DIR=$2
DATA_DIR=$DATA_ROOT_DIR/trn"$phase"_"$layerType"
echo "$DATA_DIR"

cat gpu$gpuid/$1-input_vec.txt >> $DATA_DIR/input_vec.txt
cat gpu$gpuid/$1-output_vec.txt >> $DATA_DIR/output_vec.txt

