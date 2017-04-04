#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import random
import csv

def randomizePooling(layer, new_data_shape):
	layer.name="POOLING"
	pool_param = layer.pooling_param
	pool_param
	new_kernel_size = random.randrange(1,4) # kernel_sizes of pooling are not big 
	new_stride = random.randrange(1,3) #
	new_pad = random.randrange(0,new_stride)
	pool_param.kernel_size=new_kernel_size
	pool_param.stride=new_stride
	if new_pad !=  0:
		pool_param.pad.append(new_pad)
	new_data_shape[2] = (new_hw + 2*new_pad - new_kernel_size)/new_stride + 1
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	input_vector.append(new_kernel_size)
	input_vector.append(new_stride)
	input_vector.append(new_pad)
	return input_vector
	
	
def randomizeIPLayer(layer,new_data_shape):
	layer.name="IP"
	ip_param = layer.inner_product_param
	ip_param.num_output= random.randrange(10,4097) # needs to be random  
	new_data_shape[1] = new_data_shape[1]*new_data_shape[2]*new_data_shape[2]
	new_data_shape[2] = 0 # this flattens out the data
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(ip_param.num_output)
	return input_vector

def randomizeDummyDataLayer(layer,new_data_shape):
	dummy_param=layer.dummy_data_param
	dummy_shape=dummy_param.shape
	dummy_shape[0].dim[0]=new_data_shape[0]# needs to be random
	if "x" in layer.name:
		dummy_shape[0].dim[1] = new_data_shape[1]# needs to be random
		dummy_shape[0].dim[2] = new_data_shape[2] # needs to be random
		dummy_shape[0].dim[3] = new_data_shape[2] # match height, as most training images are square sized
		

def randomizeConvLayer(layer,new_data_shape):
	layer.name="CONV"
	conv_param = layer.convolution_param
	new_output = random.randrange(3,101)
        conv_param.num_output = new_output	
	new_kernel_size = random.randrange(1,12) 
        conv_param.kernel_size[0] = new_kernel_size # needs to be random and one-dimensional
	new_stride = random.randrange(1,3)
        conv_param.stride[0] = new_stride# needs to be random and one-dimensional
	new_pad = random.randrange(0,new_stride)
	if new_pad != 0:
        	conv_param.pad.append(new_pad) #needs to be random and one-dimensional
	input_vector = [] # input_vector used for training
	new_data_shape[1] = new_output
	new_data_shape[2] = (new_hw + 2*new_pad - new_kernel_size)/new_stride + 1
	#input_vector.append(input_size)
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	input_vector.append(new_output)
	input_vector.append(new_kernel_size)
	input_vector.append(new_stride)
	input_vector.append(new_pad)
	return input_vector
	
         

def parse_args():
	parser = ArgumentParser(description=__doc__,formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('input_net_prototxt',
			help='the base prototxt to start with')
	parser.add_argument('output_prototxt',
			help='randomized prototxt file')
	parser.add_argument('layer_type',
			help='type of layer to randomize')
	parser.add_argument('input_csv_file',
			help='csv file which containes the input vectors')
	args = parser.parse_args()

	return args

def randomizeLayers(net_prototxt,layer_type):
	input_data_shape = []
	input_data_shape.append(random.randrange(2,257)) #num_of_batch
	input_data_shape.append(random.randrange(3,101)) # new num of channel
	input_data_shape.append(random.randrange(14,226)) #new hidth/width , I think it is safe to assume data is going to be given in a square shape
	for layer in net_prototxt.layer:
		if layer.type == "DummyData":
			randomizeDummyDataLayer(layer,input_data_shape)
		elif layer.type == "Convolution":			
			input_vector=randomizeConvLayer(layer,input_data_shape)
			if layer_type == "Convolution":
				return_vec = input_vector								
		elif layer.type == "InnerProduct":
			input_vector=randomizeIPLayer(layer,input_data_shape)
			if layer_type == "InnerProduct":
				return_vec = input_vector
		elif layer.type == "Pooling"
			input_vector=randomizePoolingLayer(layer,input_data_size)
			if layer_type == "Pooling":
				return_vec = input_vector
	return return_vec
		

def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    fin = open(args.input_net_prototxt,"rb") 
    text_format.Merge(fin.read(), net)
    fin.close()
    input_vector = randomizeLayers(net,args.layer_type)
    output = open(args.output_prototxt,"w")
    output.write(text_format.MessageToString(net))
    output.close()
    with open(args.input_csv_file,"a") as f:
	for item in input_vector:
	   f.write(str(item)+',')
	f.write('\n')
    
if __name__ == '__main__':
    main()
