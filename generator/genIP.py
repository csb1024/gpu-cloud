#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import random
def genIPLayer(layer):
	layer.name="IP_rand"
	layer.type="InnerProduct"
	layer.bottom.append("data")
	layer.top.append("IP1")
	layer.param.add().lr_mult=1
	layer.param.add().lr_mult=1  # not a mistake, does NOT need to be random`
	ip_param = layer.inner_product_param
	ip_param.num_output= random.randrange(10,4097) # needs to be random
	ip_param.weight_filler.type="constant" # not sure whether to randomize or not
	ip_param.bias_filler.type="constant" # this one too	
