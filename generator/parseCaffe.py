#!/usr/bin/env python
import matplotlib 
matplotlib.use('Agg') 
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

"""
** Metric was monitored in the following order 
1. dram_utilization -> line num 7
2. achieved_occupancy -> line num 8
3. sysmem_utilization -> line num 9
4. executed_ipc -> line num 10
"""

"""
**
Execution time was added to this code,2017-4-12
"""
def parseCaffeLogforTime(caffe_log,layer_name):
	exec_time=0.
	isbw = False
	foundline = False
        fw_time = 0
	bw_time = 0
	with open(caffe_log) as fp:
	    for line in fp:
	    	words = line.split()
	    	for word in words:
			if word == "[time]":
				foundline=True
	    		if foundline and word == layer_name:
				if not isbw:
					fw_time = float(words[-2])
					isbw= True
				else:
					bw_time = float(words[-2])
	    foundline = False
	exec_time = fw_time + bw_time
	return exec_time


def average(numbers):
	total = sum(numbers)
	total = float(total)
	return total / len(numbers) 
					
						
				

def parseCaffeLog(caffe_log,layer):
	layer_name = '['+layer+']' # adding big brackets to both side
	total_mem = 0.
	with open(caffe_log) as fp:
	    for line in fp:
	    	words = line.split()
	    	for word in words:
	    		if word == layer_name:
	    			total_mem = total_mem + float(words[-1])
	return total_mem

# leaving the argument function and main function for debugging
def parse_args():
	parser = ArgumentParser(description=__doc__,
			formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument('caffe_log',
			help='input caffe log file')
	parser.add_argument('layer_name',
			help='name of layer to parse')
	parser.add_argument('output_csv_file',
			help='file to store output vector')
	args = parser.parse_args()
	return args
def main():
	args = parse_args()
	result_vec = []
#	mem_allocated=parseCaffeLog(args.caffe_log, args.layer_name)	
#	result_vec = parseNvprofLog(args.layer_type,args.nvprof_log_dir)
	exec_time = parseCaffeLogforTime(args.caffe_log,args.layer_name)
#       result_vec.append(mem_allocated)
	result_vec.append(exec_time)
#	result_vec.append(mem_allocated)
#	print result_vec
		
	with open(args.output_csv_file,"a") as f:
        	for item in result_vec:
	           f.write(str(item)+',')
		f.write('\n')


if __name__ == '__main__':
	main()
