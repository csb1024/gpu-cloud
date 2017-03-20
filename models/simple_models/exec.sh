#!/bin/bash
NVML_MON=$HOME/git/gpu-cloud/monitor/cpp/nvml_mon
PYTHON="/usr/bin/python"
export CUDA_VISIBLE_DEVICES="0"

script="multilayer_perceptron.py"
echo "Execute nvml_mon"
$NVML_MON $PWD &
echo "Execute ${script}"
$PYTHON  ${script}
