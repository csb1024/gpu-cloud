#!/bin/bash

#MLP
MLP_SCRIPT=single_FC_layer.py
CNN_SCRIPT=convolutional_network.py
S_RNN_SCRIPT=recurrent_network.py
D_RNN_SCRIPT=dynamic_rnn.py 

# directory for soring logs
LOG_DIR=$PWD

# set visible GPUs to CUDA
export CUDA_VISIBLE_DEVICES=0

echo "Excuting single FC"
python ${MLP_SCRIPT} &
python ${PWD}/nvml_monitor.py ${LOG_DIR}
mv ${LOG_DIR}/monitor_log.txt ${LOG_DIR}/single_FC_log.txt

#echo "Executing CNN"
#python ${CNN_SCRIPT} &
#python ${NVML_DIR}/nvml_monitor.py
#mv ${NVML_DIR}/monitor_log.txt ${NVML_DIR}/cnn_log.txt

#echo "Executing Static RNN"
#python ${S_RNN_SCRIPT} &
#python ${NVML_DIR}/nvml_monitor.py
#mv ${NVML_DIR}/monitor_log.txt ${NVML_DIR}/s_rnn_log.txt

#echo "Executing Dynamic RNN"
#python ${D_RNN_SCRIPT} &
#python ${NVML_DIR}/nvml_monitor.py
#mv ${NVML_DIR}/monitor_log.txt ${NVML_DIR}/d_rnn_log.txt

