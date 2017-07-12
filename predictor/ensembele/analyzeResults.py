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
	parser.add_argument('ensemble_data_txt',
			help='file which contains results of ensemble model') # for validation data
	parser.add_argument('label_data_txt',
			help='file which contains the label data') # for validation labels
	parser.add_argument('results_txt',
			help='file which will contain the analyzed results')

	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	e_results = np.loadtxt(args.ensemble_data_txt)
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
