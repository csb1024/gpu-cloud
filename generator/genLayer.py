#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2
import genConv
import genIP
import genDummyData
def parse_args():
	"""
	Parse input arguments
	"""
	parser = ArgumentParser(description=__doc__,formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('output_text_file',help='Output text file')
	parser.add_argument('type',help='type of layer to produce')
	args = parser.parse_args()

	return args

def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    net.name="random_gen_single_conv_net"

    genDummyData.genDummyDataLayer(net.layer.add())
    if args.type == "Convolution":
    	genConv.genConvLayer(net.layer.add())
    elif args.type == "InnerProduct":
    	genIP.genIPLayer(net.layer.add())
    else:
    	print "unknown value for type, ",args.type
	sys.exit(1)
    output = open(args.output_text_file,"w")
    output.write(text_format.MessageToString(net))
    output.close()
    
if __name__ == '__main__':
    main()
