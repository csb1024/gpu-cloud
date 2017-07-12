from __future__ import print_function 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
import math
def returnLog(vectors):
#        print(vectors)
        new_vectors = []
        for i in range(len(vectors)):
                vector=np.log(vectors[i])
 #               print(vector)
                new_vectors.append(vector)
        return new_vectors

def parse_args():
	parser = ArgumentParser(description=__doc__,
		formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('input_file',
			help= 'input file you want to check')
	args = parser.parse_args()
	return args

def readInput(input_txt):
        vectors = []
        with open(input_txt,"rb") as csvfile:
                reader = csv.reader(csvfile,delimiter=',')
                for row in reader:
                        del row[-1]
                        narray = np.array(row,dtype=np.float)
                        vectors.append(narray)
        return vectors

def checkLEZero(vectors):
	check_flag = False
	vec_len = len(vectors)
	if vec_len == 0:
		vec_size = 0
	else:
		vec_size = len(vectors[0])

	for i in range(vec_len):
		for j in range(vec_size):
			if (vectors[i][j] <= 0):
				print(vectors[i][j])
				check_flag = True
	if (check_flag):
		print("There are values less or equal to zero")
	else:
		print("Everything is fine")

			

def main():
	args = parse_args()
	vectors = readInput(args.input_file)
#	checkLEZero(vectors)
	vectors = returnLog(vectors)
	print(vectors)
	


if __name__ == '__main__':
	main()
