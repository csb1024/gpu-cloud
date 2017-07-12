'''
Predictor MLP model for gpu-cloud
'''
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import csv
import random
import sys


def parseNVMLLog(log_file):
	with open(log_file,"rb") as fp:
		reader = csv.reader(fp,delimiter=',')
		maxMem = 0.
		i = 0
		for row in reader:
			if i ==0: #skip first line
				i = i +1
				continue
			if float(row[1]) > maxMem:
				maxMem = float(row[1])
		return maxMem
def parseSTDIOLog(log_file):
	with open(log_file, "rb") as fp:
		lines = fp.readlines()
		i =0
		total_time = 0.
		for line in lines:
			if ( i == 5): # data is on the 6th line
				words = line.split()
				total_time = float(words[2])
			i = i +1
		return total_time
		
		

def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('nvml_monitor_file',
			help='nvml monitor log file')
	parser.add_argument('stdio_log',
			help='stdio of nvml')
#	parser.add_argument('input_net_prototxt',
#			help='the input prototxt to parse')
#	parser.add_argument('checkpoint_dir',
#			help='directory where models checkpoints are') 
	args = parser.parse_args()
	return args
def main():
	args=parse_args()
	alloc_mem_percent = parseNVMLLog(args.nvml_monitor_file)
	exec_time = parseSTDIOLog(args.stdio_log)
	print exec_time
if __name__ == '__main__':
	main()
