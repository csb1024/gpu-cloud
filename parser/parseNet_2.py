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
def parsePoolingLayer(layer):

def parseDataLayer(layer):

def parseConvLayer(layer):
      if len(layer.convolution_param.stride) == 1:
	 stride = layer.convolution_param.stride[0] # only one
      else:
	 stride = min(layer.convolution_param.stride)
      # kernel_size   
      if len(layer.convolution_param.kernel_size) == 1:
	 kernel_size = layer.convolution_param.kernel_size[0] * layer.convolution_param.kernel_size[0]# only one
      else:
	 kernel_size = layer.convolution_param.kernel_size[1] * layer.convolution_param.kernel_size[2]

      conv_output.append(layer.convolution_param.num_output)
      conv_stride.append(stride)
      conv_kernel.append(kernel_size)
f len(layer.convolution_param.stride) == 1:
		    stride = layer.convolution_param.stride[0] # only one
	 else:
	    stride = min(layer.convolution_param.stride)
	 # kernel_size   
	 if len(layer.convolution_param.kernel_size) == 1:
	    kernel_size = layer.convolution_param.kernel_size[0] * layer.convolution_param.kernel_size[0]# only one
	 else:
	    kernel_size = layer.convolution_param.kernel_size[1] * layer.convolution_param.kernel_size[2]

	 conv_output.append(layer.convolution_param.num_output)
	 conv_stride.append(stride)
	 conv_kernel.append(kernel_size)



def parseLayers(net_prototxt,phase):
# list of variables(and vectors) to maintain thoughout the execution
   layer_num = 0
   batch_size = 0
#input_size = 0
   conv_num = 0
   conv_output = []
   conv_stride = []  
   conv_kernel = []
   ip_num = 0
   ip_output = []
   pool_num = 0
   pool_stride = []
   pool_kernel = []
# output_type = 0
   for layer in net_prototxt.layer:
      if layer.type == "Data":
	 for layer_phase in layer.include:
	   if layer_phase.phase == phase:
	      batch_size = layer.data_param.batch_size
	      parseDataLayer(layer)
      elif layer.type == "Convolution":
         parseConvLayer(layer)
	 # stride
     elif layer.type == "Pooling":
	 pool_num = pool_num + 1
	 pool_stride.append(layer.pooling_param.stride)
	 pool_kernel.append(layer.pooling_param.kernel_size)
	 parsePoolingLayer(layer)
      elif layer.type == "InnerProduct":
	 ip_num = ip_num + 1
	 ip_output.append(layer.inner_product_param.num_output)
    layer_num = layer_num + 1
   
# The following will change for scenario 2 
   vector = []
   vector.append(layer_num)
   vector.append(batch_size)
   vector.append(conv_num)
   vector.append(sum(conv_output))
   vector.append(min(conv_stride))
   vector.append(min(conv_kernel))
   vector.append(ip_num)
   vector.append(sum(ip_output))
   vector.append(pool_num)
   vector.append(min(pool_stride))
   vector.append(min(pool_kernel))
   return vector



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

    input_vector = parseLayers(net,phase)

    #for debugging
    i=0
    while i < len(input_vector):
    	print i+1,"th item : ",input_vector[i]
	i = i+1
    print('Printing net to %s' % args.output_text_file)
    output = open(args.output_text_file,"w")
    i=0
    while i < len(input_vector):
	string = str(input_vector[i]) + "\n"
	output.write(string)
	i = i+1

    output.close()
    
if __name__ == '__main__':
    main()
