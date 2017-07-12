#!/bin/bash


# Monitor related constants
NVML_DIR="/home/sbchoi/git/gpu-cloud/monitor/cpp"

NVML_BIN="nvml_mon"

# 
LOG_DIR=$PWD

#extra info, stdout redirected
OUTPUT="stdio.log"

#name of the log file
LOG="monitor_log.txt"

PYTHON="/usr/bin/python"

PARSER_DIR="/home/sbchoi/git/gpu-cloud/predictor/intergrator"
PARSER="nvmlParser.py"

$NVML_DIR/$NVML_BIN $LOG_DIR > $LOG_DIR/$OUTPUT

$PYTHON $PARSER_DIR/$PARSER $LOG_DIR/$LOG $LOG_DIR/$OUTPUT

