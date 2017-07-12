#!/usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import csv
import math
import numpy as np

def filterVectors(data_vectors, label_vectors):
	new_data_vectors = []
	new_label_vectors = []
	num_of_vecs = len(label_vectors)
        print "filtering "+str(num_of_vecs)
	i =0 
	for i in range(num_of_vecs):
		vec_len = len(label_vectors[i])
		containZero = False
		for j in range(vec_len):
			if label_vectors[i][j] == 0.0 or label_vectors[i][j] == 0:
				containZero=True
		if not containZero:
			# assuming data vector and label vetor have same length to each other
			new_data_vectors.append(data_vectors[i])
			new_label_vectors.append(label_vectors[i])
	# keep getting errors when returning so comining each into one list
	ret_list = []
	ret_list.append(new_data_vectors)
	ret_list.append(new_label_vectors)
	return ret_list
				

		
	

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
	parser = ArgumentParser(description=__doc__,formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('org_data_file',
			help='csv file which stores original data vectors')
	parser.add_argument('org_label_file',
			help='csv file which stores original data labels')
	parser.add_argument('new_data_file',
			help='csv file which stores new data vectors')
	parser.add_argument('new_label_file',
			help='csv file which stores new label vectors')
	args = parser.parse_args()

	return args
	

def main():
    args = parse_args()
    org_data_vector = readInput(args.org_data_file)
    org_label_vector  = readInput(args.org_label_file)
    temp = filterVectors(org_data_vector,org_label_vector)
    new_data_vector = temp[0]
    new_label_vector = temp[1]
    print str(len(new_data_vector)) + "data remained after filtering" 
    with open(args.new_data_file,"w") as f:
    	for vector in new_data_vector:
		for item in vector:
	   		f.write(str(item)+',')
		f.write('\n')

    with open(args.new_label_file,"w") as f:
    	for vector in new_label_vector:
		for item in vector:
	   		f.write(str(item)+',')
		f.write('\n')
    
if __name__ == '__main__':
    main()
