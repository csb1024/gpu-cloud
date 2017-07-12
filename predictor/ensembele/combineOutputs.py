'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import numpy as np
import csv
import random

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
	parser.add_argument('trn_data_txt',
			help='file which contains training data vectors') # for data
	parser.add_argument('trn_label_txt',
			help='file which contains training label vectors') # for label
	parser.add_argument('val_data_txt',
			help='file which contains validation data vectors') # for validation data
	parser.add_argument('val_label_txt',
			help='file which contains validation label vectors') # for validation labels
	parser.add_argument('checkpoint_dir',
			help='directory where checkpoint files will be stored/loaded, pass "new" for training new parameters')
	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	trn_data = readInput(args.trn_data_txt) # read input data
	trn_label = readInput(args.trn_label_txt)
	val_data = readInput(args.val_data_txt)
			np.savetxt('input_vec.txt',input_vec,delimiter=',')
		np.savetxt('label_vec.txt',label_vec,delimiter=',')
		np.savetxt('output_vec.txt',output_vec,delimiter=',')

		# Test Accuracy against validation data
	
if __name__ == '__main__':
	main()
