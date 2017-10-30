#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

"""
The input vectors of Scenario 2
<Conv Layers>

<Data Layer>

<Common Layer - Innerproduct>

<Dropout>


"""
def parsePoolingLayer(layer, new_data_shape):
	layer.name="POOLING"
	pool_param = layer.pooling_param
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
#	if len(pool_param.pad) == 0:
#		pad_size = 0
#	else:
#		pad_size = pool_param.pad[0]

#	if len(pool_param.stride == 0):
#		stride_size = pool_param.stride
#	else:
	
	input_vector.append(pool_param.pad)	
	input_vector.append(pool_param.kernel_size)
	input_vector.append(pool_param.stride)
	#reshaping for output
	new_data_shape[2] = int(new_data_shape[2] + 2*pool_param.pad - pool_param.kernel_size)/pool_param.stride + 1
	return input_vector

def parseReLULayer(layer,new_data_shape):
	input_vector = []
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	return input_vector

def parseDataLayer(layer):
	new_data_shape = []
	trans_param = layer.transform_param
	dat_param = layer.data_param
	
	
	# batch
	new_data_shape.append(dat_param.batch_size)
	
	# channel
	if "imagenet" in dat_param.source: 
		new_data_shape.append(3)
	else:
		new_data_shape.append(5)

	# h/w (square sized)
	new_data_shape.append(trans_param.crop_size)
	# input_vector : batch, cropsize
	return new_data_shape

def parseConvLayer(layer,new_data_shape):
	conv_param = layer.convolution_param
	
	input_vector = [] # input_vector used for training
	input_vector.append(new_data_shape[0])
	input_vector.append(new_data_shape[1])
	input_vector.append(new_data_shape[2])
	#reshaping for output
	new_data_shape[1] = conv_param.num_output
	if len(conv_param.pad) == 0:
		pad_size = 0
	else:
		pad_size = conv_param.pad[0]
	if (len(conv_param.stride) == 0):
		stride_size = 1
	else:
		stride_size = conv_param.stride[0]
	new_data_shape[2] = int(new_data_shape[2] + 2*pad_size - conv_param.kernel_size[0])/stride_size + 1
	
	# making input vector
	input_vector.append(conv_param.num_output)
	input_vector.append(conv_param.kernel_size[0])
	input_vector.append(stride_size)
	input_vector.append(pad_size)
	return input_vector
def parseLSTMLayer(layer,new_data_shape):
	input_vector = []
	params = layer.recurrent_param

	# dimension of data and label need to match 
	new_num_output = new_data_shape[2]
	# assume that data is a 1-d vector
	input_vector.append(new_data_shape[0]) # len/stream
	input_vector.append(new_data_shape[1]) # num of streamsa
	input_vector.append(new_data_shape[2]) # data size
	input_vector.append(new_num_output)

	# resulting data
	# new_data_shape[0] and new_data_shape[1] are unchanged
	# since label = dimension data, new_data_shape[2] also does not change
		
	return input_vector

def parseIPLayer(layer,new_data_shape):
	vector = []
	ip_param = layer.inner_product_param
	# reshaping input
	new_data_shape[1] = new_data_shape[1]*new_data_shape[2]*new_data_shape[2]
	new_data_shape[2] = 1 # this flattens out the data

	vector.append(new_data_shape[0])
	vector.append(new_data_shape[1])
	vector.append(ip_param.num_output)
	#reshaping for output
	new_data_shape[1] = ip_param.num_output
	return vector
def parseLayers(net_prototxt,phase):

   layer_type = []# vector of layer types(string)
   layer_info = []#vector of layer info (vectors)
# list of variables(and vectors) to maintain thoughout the execution
   input_data =[] # must follow the caffe convention of defining data type
   

# input_vectors 
   
# output_type = 0
   for layer in net_prototxt.layer:
      if layer.type == "Data": # used for parsing the initial data size
	 for layer_phase in layer.include:
	   if layer_phase.phase == phase:
	      batch_size = layer.data_param.batch_size
	      next_layer_input=parseDataLayer(layer)
	      		
      elif layer.type == "Convolution":
         vector=parseConvLayer(layer,next_layer_input)
	 layer_type.append(layer.type)
	 layer_info.append(vector)

      elif layer.type == "Pooling":
	 vector=parsePoolingLayer(layer,next_layer_input)
	 layer_type.append(layer.type)
	 layer_info.append(vector)
	 
      elif layer.type == "InnerProduct":
	 vector=parseIPLayer(layer,next_layer_input)
	 layer_type.append(layer.type)
	 layer_info.append(vector)
      elif layer.type == "ReLU":
	 vector=parseReLULayer(layer,next_layer_input)
	 layer_type.append(layer.type)
	 layer_info.append(vector)
      elif layer.type == "LSTM":
	 vector = parseLSTMLayer(layer,next_layer_input)
	 layer_type.append(layer.type)
	 layer_info.append(vector)
	  

   return layer_type, layer_info



def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_text_file',
                        help='Output text file')
    parser.add_argument("phase",
		       help="the phase, chose between TRAIN and TEST")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.phase == "TRAIN":
	phase = caffe.TRAIN
    elif args.phase == "TEST":
	phase = caffe.TEST
    else:
	print "unknown value for phase, ", args.phase
	sys.exit(1)
    net = caffe_pb2.NetParameter()
    fin = open(args.input_net_proto_file,"rb")    
    text_format.Merge(fin.read(), net)
    fin.close()

    [layer_types, layer_infos] = parseLayers(net,phase)

    #for debugging
	
    for name in layer_types:
    	print name
    print('Printing net to %s' % args.output_text_file)
    output = open(args.output_text_file,"w")
    for vector in layer_infos:
	print vector
	string=""
	for item in vector:
		string = string+str(item) + ","
	output.write(string+"\n")
    output.close()
    
if __name__ == '__main__':
    main()
