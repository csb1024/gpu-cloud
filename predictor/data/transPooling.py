'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('org_data_txt',
			help='file which contains results of ensemble model') # for validation data
	parser.add_argument('new_data_txt',
			help='file which contains the label data') # for validation labels

	args = parser.parse_args()
	return args

def storeNormData(norm_info_file, norm_vectors):
	with open(norm_info_file,"w") as fp: 
		for vector in norm_vectors:
			for item in vector:
				fp.write(str(item)+',')
			fp.write('\n')

	
def readInput(input_txt):
	vectors = []
	with open(input_txt,"rb") as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
		del row[-1]
		narray = np.array(row,dtype=np.float)
		vectors.append(narray)
	return vectors



def main():
	args=parse_args()
	e_results = readInput(args.org_data_txt)
	labels = np.loadtxt(args.label_data_txt)
#	print (labels)

	error = []
	strs = []
	for i in range(len(labels)):
		error.append(abs(e_results[i] - labels[i])/labels[i]*100)
		t_str = str(e_results[i]) +","+str(labels[i])+","+str(error[i])
		strs.append(t_str)
#	np.savetxt(args.results_txt,input_vec,delimiter=',')
	fout = open(args.results_txt,"w")
	fout.write("Ensembled Results,Labels,Error(%)\n")
	for i in range(len(labels)):
		fout.write(strs[i])
		fout.write("\n")

		

		
		
	
if __name__ == '__main__':
	main()
