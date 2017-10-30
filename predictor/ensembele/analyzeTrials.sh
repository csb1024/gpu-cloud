#!/bin/bash

if [ -z "$1" ]
then
echo "Please spefify the GPU number Ex) 0 ~ 3"
exit
fi

if [ -z "$2" ]
then
echo "Please specify number of trials"
exit
fi
PYTHON="/usr/bin/python" # dir to python
CODE=combineResults.py
GPU_CLOUD_ROOT="/home/sbchoi/git/gpu-cloud"
GPU_ID=$1
LOG_DIR=$PWD/gpu$GPU_ID
TRIAL=$2
analysis_dir=$PWD/results

# erase previous analysis
rm $analysis_dir/*

for (( a=1; a<=${TRIAL}; a++));do

cp $LOG_DIR/$a-results.txt $analysis_dir 
done

$PYTHON $CODE $analysis_dir
