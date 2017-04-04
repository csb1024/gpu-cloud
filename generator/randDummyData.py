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

def randomizeLayers(net_prototxt):
   for layer in net_prototxt.layer:
      
      if layer.type == "Convolution":
      	layer.convolution_param.num_output = 600
	


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('outprototxt',
                        help='Output prototxt file')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    fin = open(args.input_net_proto_file,"rb")    
    text_format.Merge(fin.read(), net)
    fin.close()

    randomizeLayers(net)
    output = open(args.outprototxt,"w")
    output.write(text_format.MessageToString(net))
    output.close()

   
    
if __name__ == '__main__':
    main()
