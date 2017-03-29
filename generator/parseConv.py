#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
def parseConvLayer(layer,prev_output):
      #input vector
      vector = []
      #parameters for input vector 
      stride = 0
      kernel_size = 0
      output_size = 0
      padding = 0
      input_size = prev_output
     
      # stride
      if len(layer.convolution_param.stride) == 1:
	 stride = layer.convolution_param.stride[0] # only one
      else:
	 stride = min(layer.convolution_param.stride) # choose the smaller one, strides are usually a single value
      
      # kernel_size   
      if len(layer.convolution_param.kernel_size) == 1:
	 kernel_size = layer.convolution_param.kernel_size[0] * layer.convolution_param.kernel_size[0]# only one
      else:
	 kernel_size = layer.convolution_param.kernel_size[1] * layer.convolution_param.kernel_size[2]

      # padding 
      if len(layer.convolution_param.pad) == 0: # no padding
         padding = 0
      elif len(layer.convolution_param.pad) == 1:
         padding = layer.convolution_param.pad[0] * layer.convolution_param.pad[0]
      else:
         padding = layer.convolution_param.pad[1] * layer.convolution_param.pad[2]      
      
      output_size = layer.convolution_param.num_output
      vector.append(stride)
      vector.append(kernel_size)
      vector.append(padding)
      vector.append(output_size)
      vector.append(input_size)
      # needs to be changed to the 'real' size in the future
      next_layer_input = output_size
      return [vector, next_layer_input]


def parseLayers(net_prototxt,phase):
# list of variables(and vectors) to maintain thoughout the execution
   batch_size = 0
   next_layer_input=1024 # arbitrary input for now, should be changed in the future
#input_size = 0
# output_type = 0
   for layer in net_prototxt.layer:
       if layer.type == "Data":
          for layer_phase in layer.include:
             if layer_phase.phase == phase:
	         batch_size = layer.data_param.batch_size
       elif layer.type == "Convolution":
          [vector, next_layer_input]=parseConvLayer(layer,next_layer_input)
   
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
    print('Printing layer vector to %s' % args.output_text_file)
    output = open(args.output_text_file,"w")
    i=0
    while i < len(input_vector):
	string = str(input_vector[i]) + "\n"
	output.write(string)
	i = i+1

    output.close()
    
if __name__ == '__main__':
    main()
