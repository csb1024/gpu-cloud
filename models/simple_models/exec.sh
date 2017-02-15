#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

script="multilayer_perceptron.py"

echo " Execute ${script}"
python  ${script}
