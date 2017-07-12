#!/bin/bash 

TEST_PROTOTXT="out.prototxt"
PYTHON="/usr/bin/python"
PARSER="predictor.py"
TESTER="inferTFE.py"


CUR_DIR=$PWD


$PYTHON $PARSER $CUR_DIR $TEST_PROTOTXT
