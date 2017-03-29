#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

def parseCaffeLog(caffe_log, layer):
    fin = open(caffe_log,"r")
    layer_name = '['+layer+']' # adding big brackets to both side
    total_mem = 0
    with open(caffe_log) as fp:
	    for line in fp:
	    	words = line.split()
	    	for word in words:
	    		if word == layer_name:
	    			total_mem = total_mem + int(words[-1])
    fin.close()
    return total_mem




# leaving the argument function and main function for debugging
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('caffe',
			help='input caffe log file')
	parser.add_argument('layer',
			help='name of layer to parse')
	args = parser.parse_args()
	return args
def main():
	args = parse_args()
	result=parseCaffeLog(args.caffe, args.layer)
	print result

if __name__ == '__main__':
	main()
