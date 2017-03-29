#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import random
def genConvLayer(layer):
	layer.name="Conv_ran"
	layer.type="Convolution"
	layer.bottom.append("data")
	layer.top.append("conv1")
	layer.param.add().lr_mult=1
	layer.param.add().lr_mult=1 # not a mistake
	conv_param = layer.convolution_param
	conv_param.num_output= random.randrange(3,101)
	conv_param.kernel_size.append(random.randrange(1,12)) # needs to be random and one-dimensional
	conv_param.stride.append(random.randrange(1,3))# needs to be random and one-dimensional
	conv_param.pad.append(random.randrange(1,3)) #needs to be random and one-dimensional
	conv_param.weight_filler.type="constant" # not sure whether to randomize or not
	conv_param.bias_filler.type="constant" # this one too	
