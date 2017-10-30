#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import random
import csv
import math

def randomizePoolingLayer(layer, new_data_shape):
	layer.name="POOLING"
	pool_param = layer.pooling_param
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	new_kernel_size = random.randrange(2,4) # kernel_sizes of pooling are not big and diverse
	new_stride = random.randrange(1,new_kernel_size) #
	new_pad = random.randrange(0,new_stride)


	if new_pad != 0:
        	pool_param.pad=new_pad #needs to be random and one-dimensional
	input_vector.append(new_pad)
	
	input_vector.append(new_kernel_size)
	input_vector.append(new_stride)
	
	pool_param.kernel_size=new_kernel_size
	pool_param.stride=new_stride
	#reshaping for output
	new_data_shape[2] = int(new_data_shape[2] + 2*new_pad - new_kernel_size)/new_stride + 1
	return input_vector
	
	
def randomizeIPLayer(layer,new_data_shape,relu_flag):
	layer.name="IP"
	if relu_flag :
		layer.name="IP2"
	ip_param = layer.inner_product_param
	#used to cap maximum number of random generation 
	rand_max = 4097 
	#caffe's int_max : 2147483647
	# reshaping input
	new_data_shape[1] = new_data_shape[1]*new_data_shape[2]*new_data_shape[2]
	new_data_shape[2] = 1 # this flattens out the data
	if (new_data_shape[0] * new_data_shape[1] > 2147483647):
#		print(new_data_shape[0] * new_data_shape[1],"!!!")
		rand_limit=2
	
	elif (new_data_shape[0] * new_data_shape[1] == 0):
		rand_limit=2
	else:
#		print(new_data_shape[0] * new_data_shape[1])
		rand_limit = int(2147483647/(new_data_shape[0]*new_data_shape[1]))

	if (rand_limit < rand_max):
		rand_max = rand_limit
	if (rand_max == 1):
		rand_max = 2

#	if relu_flag :
#		ip_param.num_output = new_data_shape[0]
#	else:		
	ip_param.num_output= random.randrange(1,rand_max) # needs to be random  
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(ip_param.num_output)
	#reshaping for output
	new_data_shape[1] = ip_param.num_output
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
	new_kernel_size = random.randrange(2,12) 
        conv_param.kernel_size[0] = new_kernel_size # needs to be random and one-dimensional
	new_stride = random.randrange(1,new_kernel_size)
        conv_param.stride[0] = new_stride
# needs to be random and one-dimensional
	new_pad = random.randrange(0,new_stride)
	if new_pad != 0:
        	conv_param.pad.append(new_pad) #needs to be random and one-dimensional
	input_vector = [] # input_vector used for training
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	#reshaping for output
	new_data_shape[1] = new_output
	new_data_shape[2] = int(new_data_shape[2] + 2*new_pad - new_kernel_size)/new_stride + 1
	
	# making input vector
	input_vector.append(new_output)
	input_vector.append(new_kernel_size)
	input_vector.append(new_stride)
	input_vector.append(new_pad)
	return input_vector
	
def randomizeReLULayer(layer,new_data_shape):
	input_vector = []
	layer.name="RELU"
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	return input_vector
def randomizeSoftmaxLossLayer(layer,new_data_shape):
	input_vector = []
	layer.name="SOFTMAX"
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	return input_vector

def randomizeLRNlayer(layer, new_data_shape):
	input_vector = []
	layer.name="LRN" 
	params = layer.lrn_param
        input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	# there are not many settings for lrn layers
	new_local_size = 5
	new_alpha = 0.0001
	new_beta=0.75

	params.local_size = new_local_size
	params.alpha = new_alpha
	params.beta = new_beta
 	params.norm_region = 0
	return input_vector	
def randomizeDataLayer(layer, new_data_shape):
	input_vector = []
	layer.name = "IMAGEDATA"
	trans_param = layer.transform_param
	dat_param = layer.data_param

 	new_crop_size = random.randrange(224, 230)
	# overwrite randomized shapes
	# batches are already randomized
	dat_param.batch_size = new_data_shape[0]
	trans_param.crop_size = new_crop_size
	new_data_shape[1] = 3
	new_data_shape[2] = new_crop_size
	# input_vector : batch, cropsize
	input_vector.append(new_data_shape[0])
	for layer_phase in layer.include:
		if layer_phase.phase == caffe.TRAIN:
			input_vector.append(new_crop_size)
	return input_vector
def randomizeDropoutLayer(layer, new_data_shape):
	input_vector = []
	layer.name = "DROP"
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
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
	relu_flag=False # specially used for generating relu data
	input_data_shape = []
	# batches are normally a power of 2 
	random_batchs = [ 16, 32, 64, 128, 256 ]
	rand_batch = set(random_batchs)
 	rand_sample= random.sample(rand_batch, 1)
	new_batch = int(rand_sample[0])
	input_data_shape.append(new_batch) #num_of_batch
	limit = int(2147483647 / new_batch)
	channel_max = 257
	if (limit < channel_max):
		channel_mx = limit
	new_channel = random.randrange(1,channel_max)
	input_data_shape.append(new_channel)
	limit  = int (math.sqrt(2147483647.0 / float(new_batch * new_channel)))
	
	size_max = 226
	if (limit < size_max):
		size_max = limit
	new_size = random.randrange(24,size_max)
#	print(channel_max)
#	print(size_max)
	input_data_shape.append(random.randrange(new_size)) #new height/width , I think it is safe to assume data is going to be given in a square shape
	for layer in net_prototxt.layer:
		if layer.type == "DummyData":
			# just fix the dummy data shape for Dropout
			if layer_type == "Dropout":
				input_data_shape[1]=3
				input_data_shape[2]=24			
			randomizeDummyDataLayer(layer,input_data_shape)
		elif layer.type == "Convolution":			
			input_vector=randomizeConvLayer(layer,input_data_shape)
			if layer_type == "Convolution":
				return_vec = input_vector								
		elif layer.type == "InnerProduct":
			if layer_type == "Dropout":
				# this must be changed if dropout is going to be tested on random cases
				new_output = random.randrange(4090,4100)
				layer.inner_product_param.num_output = new_output
				input_data_shape[1]=new_output
				input_data_shape[2]=1
			else:										
				input_vector=randomizeIPLayer(layer,input_data_shape,relu_flag)
			if layer_type == "InnerProduct":
				return_vec = input_vector
		elif layer.type == "Pooling":
			input_vector=randomizePoolingLayer(layer,input_data_shape)
			if layer_type == "Pooling":
				return_vec = input_vector
		elif layer.type == "ReLU":
			input_vector=randomizeReLULayer(layer,input_data_shape)
			if layer_type == "ReLU":
				relu_flag=True
				return_vec = input_vector
		elif layer.type == "SoftmaxWithLoss":
			input_vector=randomizeSoftmaxLossLayer(layer,input_data_shape)
			
			if layer_type == "SoftmaxWithLoss":
				return_vec = input_vector
		elif layer.type == "LRN":
			input_vector=randomizeLRNlayer(layer,input_data_shape) 
			if layer_type == "LRN":
				return_vec = input_vector
		elif layer.type == "Data":
			input_vector = randomizeDataLayer(layer, input_data_shape)
			if layer_type == "Data" and len(input_vector) == 2:
				return_vec = input_vector
		elif layer.type == "Dropout":
			input_vector = randomizeDropoutLayer(layer, input_data_shape)
			if layer_type == "Dropout":
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
