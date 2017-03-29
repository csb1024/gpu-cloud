#!/bin/bash
set -e 
CAFFE_ROOT='/home/sbchoi/git/caffe'

$CAFFE_ROOT/build/tools/caffe train --solver=lenet_solver.prototxt 
