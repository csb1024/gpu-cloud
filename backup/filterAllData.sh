#!/bin/bash

if [ -z $1 ] 
then
echo "Please Specify the prefix of ***-input_vec.txt"
exit
fi

prefix=$1

PYTHON="/usr/bin/python"
STORAGE_DIR="/home/sbchoi/git/gpu-cloud/backup"
FINAL_DATA_DIR=$STORAGE_DIR/local-$prefix
DATA_FILE=$prefix-input_vec.txt 
LABEL_FILE=$prefix-output_vec.txt
SCRIPT=filterData.py

mkdir -p $FINAL_DATA_DIR
for i in `seq 0 3`
do
data_dir=$STORAGE_DIR/"gpu"$i
cd $data_dir
$PYTHON $STORAGE_DIR/$SCRIPT $DATA_FILE $LABEL_FILE temp_input.txt temp_output.txt
mv temp_input.txt $DATA_FILE
mv temp_output.txt $LABEL_FILE
cat $data_dir/$DATA_FILE >> $FINAL_DATA_DIR/$DATA_FILE
cat $data_dir/$LABEL_FILE >> $FINAL_DATA_DIR/$LABEL_FILE
done
echo "All data and label is stored in "$FINAL_DATA_DIR" each file is "$DATA_FILE" and "$LABEL_FILE


