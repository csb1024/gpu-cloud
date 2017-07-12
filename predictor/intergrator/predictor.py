'''
Predictor MLP model for gpu-cloud
'''
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
import random
import caffe
import sys
from caffe.proto import caffe_pb2 
from google.protobuf import text_format
import inferTFE
import combineOutput

'''
expected form of output vector from each layer
(In other words, expected input from each layer)
1. memory allocation 
2. exec time
--> all elements above require summation
3. DRAM Utilization
4. Occupancy
5. 
'''
def writeToFile_1d(dir_file, vector):
	with open(dir_file,"w") as fp:
		for item in vector:
			fp.write(str(item)+",")
		fp.write('\n')

def writeToFile_2d(dir_file, vectors):
	with open(dir_file,"w") as fp:
		for vector in vectors:
			for item in vector:
				fp.write(str(item)+",")
			fp.write('\n')
def parseInput(net, input_data_shape):


#Returns whether input shape was given as input_shape blob
#If so, fill input_data_shape 
	if(len(net.input_shape)): 
		for shape in net.input_shape:
			input_data_shape.append(shape.dim[0])
			input_data_shape.append(shape.dim[1])
			input_data_shape.append(shape.dim[2])	
# input shape is defined within layer
	else:
		for layer in net.layer:
			if layer.type == "Input":
#				print layer.input_param.shape
				for shape in layer.input_param.shape:
					input_data_shape.append(shape.dim[0])
					input_data_shape.append(shape.dim[1])
					input_data_shape.append(shape.dim[2])
			elif layer.type == "Data":
				# still need to work on it
				# need to fine data size for given backend				
#				print layer.include
				for layer_phase in layer.include:
				  # for now considering TRAIN phase only
				  if layer_phase.phase == caffe.TRAIN:
					input_data_shape.append(layer.data_param.batch_size)		      
					input_data_shape.append(3)
					input_data_shape.append(layer.transform_param.crop_size)
			elif layer.type == "DummyData":
			  
			  for layer_phase in layer.include:
			    print layer_phase
			    if layer_phase.phase == caffe.TRAIN:
			      for shape in layer.dummy_data_param.shape:
			        if(len(input_data_shape) == 0): # met dummy data first
				  input_data_shape.append(shape.dim[0])
				  input_data_shape.append(shape.dim[1])
				  input_data_shape.append(shape.dim[2])



def parseDataLayer(net, new_data_shape):
	new_data_shape.append(10)
	new_data_shape.append(3)
	new_data_shape.append(20)



def parseConvLayer(layer, new_data_shape):
#	layer.name="CONV"
	conv_param = layer.convolution_param
	input_vector = [] # input_vector used for training
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	new_data_shape[1] = conv_param.num_output
	if(len(conv_param.pad)):
		conv_pad = conv_param.pad[0]
	else:
	 	conv_pad = 0
	 
	if(len(conv_param.stride)):
		conv_stride = conv_param.stride[0]
	else:
		conv_stride = 1
	
	conv_kernel_size = conv_param.kernel_size[0]
	
	new_data_shape[2] = (new_data_shape[2] + 2*conv_pad - conv_kernel_size)/conv_stride + 1
	input_vector.append(conv_param.num_output)
	input_vector.append(conv_kernel_size)
	input_vector.append(conv_stride)
	input_vector.append(conv_pad)
	return input_vector


def parseReLULayer(layer, new_data_shape):
	input_vector = []
#	layer.name="RELU"
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	return input_vector


def parseIPLayer(layer, new_data_shape):
#	layer.name="IP"
	input_vector = []
	ip_param = layer.inner_product_param
	

	new_data_shape[1] = new_data_shape[1]*new_data_shape[2]*new_data_shape[2]
	new_data_shape[2] = 1 # this flattens out the data
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(ip_param.num_output)
	new_data_shape[1]=ip_param.num_output
	return input_vector


def parsePoolingLayer(layer, new_data_shape):
#	layer.name="POOLING"
	pool_param = layer.pooling_param
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	if (pool_param.pad):
		pool_pad = pool_param.pad
	else:
		pool_pad = 0
	input_vector.append(pool_pad)
	input_vector.append(pool_param.kernel_size)
	input_vector.append(pool_param.stride)
	new_data_shape[2] = (new_data_shape[2] - pool_param.kernel_size+2*pool_pad)/pool_param.stride + 1
	return input_vector

def parseSoftmaxLayer(layer, new_data_shape):
	input_vector = []
#	layer.name="SOFTMAX"
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	return input_vector


def parsePrototxt(net_prototxt):
	input_vectors = []
        input_data_shape = []
	network_struct = []
	parseInput(net_prototxt,input_data_shape)
	for layer in net_prototxt.layer:
		print layer.name	
	        if layer.type == "Convolution":
			input_vector=parseConvLayer(layer,input_data_shape)
			network_struct.append(layer.type)
			input_vectors.append(input_vector)
		elif layer.type == "InnerProduct":		
			input_vector=parseIPLayer(layer,input_data_shape)
			network_struct.append(layer.type)
			input_vectors.append(input_vector)
		elif layer.type == "Pooling":
			input_vector=parsePoolingLayer(layer,input_data_shape)	
			network_struct.append(layer.type)
			input_vectors.append(input_vector)
		elif layer.type == "ReLU":
			input_vector=parseReLULayer(layer,input_data_shape)
			network_struct.append(layer.type) 
			input_vectors.append(input_vector)
		elif layer.type == "SoftmaxWithLoss":
			input_vector=parseSoftmaxLayer(layer,input_data_shape)
			network_struct.append(layer.type)
			input_vectors.append(input_vector)

#	print input_vectors
#	print network_struct
	return input_vectors, network_struct


def readInput(input_txt):
	vectors = []
	with open(input_txt,"rb") as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			del row[-1]
			narray = np.array(row,dtype=np.float)
			vectors.append(narray)
	return vectors

def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('output_dir',
			help='dir where you will store predicted results')
	parser.add_argument('input_net_prototxt',
			help='the input prototxt to parse')
#	parser.add_argument('checkpoint_dir',
#			help='directory where models checkpoints are') 
	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	net = caffe_pb2.NetParameter()
	fin = open(args.input_net_prototxt,"rb")
	text_format.Merge(fin.read(), net)
	fin.close()
	input_vectors, net_struct = parsePrototxt(net)
#	writeToFile(args.meta_dir+"/input_list")
#	writeToFile(args.metat_dir+"/type_list")
	print "Parsed Prototxt file : "+args.input_net_prototxt
	print input_vectors
	print net_struct
	data_dir="/home/sbchoi/git/gpu-cloud/predictor/"
	
	#Need to hard code the following types  
	output_types = []
#	output_types.append("allocated_mem")
	output_types.append("exec-time")


	output_vec_size = len(output_types)
	# the number of ensemble models to be used for testing

	en_num = 5
	layer_wise_vectors = []
	for i  in range(len(input_vectors)):
		#need a numpy wrapper for vectors
		param_input_array = []	
		narray = np.array(input_vectors[i],dtype=np.float)
		print narray
		param_input_array.append(narray)
		
		output_vector = inferTFE.predictPerf(param_input_array, output_vec_size,en_num,net_struct[i])
		layer_wise_vectors.append(output_vector)
	
#	writeToFile(args.output_dir+"",output_vectors)
	print layer_wise_vectors
	pred_vector =combineOutput.combineResults(layer_wise_vectors, output_types)
	print pred_vector
	writeToFile_1d(args.output_dir +  "/pred_results.txt", pred_vector)
	



	
#	np.savetxt('test1.txt',input_vectors,delimiter=',')
#	np.savetxt('test2.txt',net_struct,delimiter=',')

#	trn_data = readInput(args.trn_data_txt) # read input data
#	trn_label = readInput(args.trn_label_txt)
#	val_data = readInput(args.val_data_txt)
#	np.savetxt('input_vec.txt',input_vec,delimiter=',')
#	np.savetxt('label_vec.txt',label_vec,delimiter=',')
#	np.savetxt('output_vec.txt',output_vec,delimiter=',')

	# Test Accuracy against validation data
	
if __name__ == '__main__':
	main()
