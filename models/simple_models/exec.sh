#!/bin/bash
#NVML_MON=$HOME/git/gpu-cloud/monitor/cpp/nvml_mon
#PYTHON="/usr/bin/python"

if [ -z "$1" ]
then
echo "Please specify how much seconds you want to wait"
exit
fi


export CUDA_VISIBLE_DEVICES="0"

script="conv_net.py"
#echo "Execute nvml_mon"
#$NVML_MON $PWD &
echo "Execute 4 ${script}"
python  ${script} &
sleep $1
python  ${script} &
sleep $1
python  ${script} &
sleep $1
python ${script} &
wait

