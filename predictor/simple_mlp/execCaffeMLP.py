#!/usr/bin/env python
"""
setup and execute training/test process for simple mlp model
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

from caffe.proto import caffe_pb2
import caffe
import numpy as np
import csv 
import sys
import h5py
def writeHDF5(data_vectors, label_vectors, data_type):
	if data_type == "Convolution":
		print "Creating HDF5 for "+data_type
	elif data_type == "InnerProduct":
		print "Creating HDF5 for "+data_type
	else:
		print "Unknown data type : "+data_type
		sys.exit(1)
	num_elem = len(data_vectors)
	in_elem = len(data_vectors[0])
	out_elem = len(label_vectors[0])
	print "Writing "+str(num_elem)+" elements"
	# store data
	data_file = h5py.File(data_type+'-data.txt','w')
	data_dataset = data_file.create_dataset(,)
	data_dataset[...] = data_vectors
	# store label


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
	parser.add_argument('input_vec_txt',
			help='file which contains input vectors') # for data
	parser.add_argument('output_vec_txt',
			help='file which containes output vectors') # for label
	parser.add_argument('data_type',
			help='the type of data you want to train for Ex) Convolution, Network')
	args = parser.parse_args()
	return args


def main():
	args=parse_args()
	input_data = readInput(args.input_vec_txt) # read input data
	input_label = readInput(args.output_vec_txt)
	writeHDF5(input_data, input_label, args.data_type)
	# execute Caffe
	#write results

	   
if __name__ == '__main__':
    main()
