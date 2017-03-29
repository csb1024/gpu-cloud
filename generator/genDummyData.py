#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import random
def genDummyDataLayer(layer):
	layer.name="DummyData_rand"
	layer.type="DummyData"
	layer.top.append("data")
	dummy_param=layer.dummy_data_param
	dummy_shape=dummy_param.shape.add()
	dim_num=random.randrange(2,257) # needs to be random
	dim_channel=random.randrange(3,101) # needs to be random
	dim_height=random.randrange(24,225) # needs to be random
	dim_width=dim_height # match height, as most training images are square sized
	if dim_num != 0:
		dummy_shape.dim.append(dim_num)
	if dim_channel != 0: # if channel is not 0 then append also height and width
		dummy_shape.dim.append(dim_channel)
		dummy_shape.dim.append(dim_height)
		dummy_shape.dim.append(dim_width)
#dummy_param.data_filler.type="constant"
