'''
Predictor MLP model for gpu-cloud
'''

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
import os
geomean = lambda n: reduce(lambda x,y: x*y, n) ** (1.0 / len(n))
def main():
	
	RESULT_DIR="/home/sbchoi/git/gpu-cloud/predictor/ensembele/results" # dir that stores results 
#	print (labels)
	err_avgs = []
	err_ids = []
	for f in os.listdir(RESULT_DIR):
		with open(RESULT_DIR+"/"+f,"r") as fp:
			err_id=f[:-12]
			print (err_id)
			errs = []
			lines = fp.readlines()
			num_of_err = len(lines)
#			print (num_of_err)
			first_line=True #flag for skipping first line
			for line in lines:
#				print (line)
				if(first_line):
					first_line=False
					continue
				words = line.split(',')
				error=float(words[-1])
				
				errs.append(error)
		summed=sum(errs)
		err_avg = float(summed/num_of_err)
#		err_avg = (geomean(errs))
		
		print (err_avg)
		err_avgs.append(err_avg)
		err_ids.append(err_id)
		
	fout = open("combined_results.txt","w")
	fout.write("Err = (Predicted - Label)/label * 100, Arith Average\n")

	for i in range(len(err_avgs)):
		err = err_avgs[i]
		fout.write(err_ids[i]+", ")
		fout.write(str(err))
		fout.write("\n")

		
		
	
if __name__ == '__main__':
	main()
